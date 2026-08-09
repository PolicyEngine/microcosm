"""The published-dataset registry: ``(country, year, variant) -> DatasetSpec``.

This is the single source of truth for what microcosm has published and where it
lives. Adding a dataset is one :class:`DatasetSpec` entry here plus its uploaded
artifact on the Hub — never a new package or repository. The loader
(:mod:`microcosm.data.loader`) reads this registry; nothing else hard-codes a
repo id, filename, or engine class.

A spec is deliberately data-only (no logic): it names the Hugging Face dataset
repo and file that hold the population, and the policyengine engine class that
reads it. The loader imports that engine lazily, so installing
``microcosm-data`` pulls neither torch nor any country engine until a load
actually needs one — country engines are optional extras (``microcosm-data[us]``).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DEFAULT_VARIANT", "DatasetSpec", "REGISTRY", "register"]

DEFAULT_VARIANT = "compact"


@dataclass(frozen=True)
class DatasetSpec:
    """A pointer to one published microcosm population and how to read it.

    Attributes:
        country: ISO-style country slug (lowercase), e.g. ``"us"``. The first
            element of the registry key and the name of the install extra that
            provides the engine (``microcosm-data[<country>]``).
        year: The population year. The second element of the registry key.
        variant: Dataset scale/contract variant. ``"compact"`` is the default
            fast national microsimulation artifact; local-geography builds
            should use a separate variant such as ``"local"``.
        hf_repo: The Hugging Face **dataset** repo holding the artifact, e.g.
            ``"policyengine/populace-us"``.
        filename: The artifact filename within ``hf_repo``, e.g.
            ``"populace_us_2024.h5"``.
        engine_module: The importable module holding the dataset class, e.g.
            ``"policyengine_us.data"``.
        engine_class: The class in ``engine_module`` that reads the H5 and is
            accepted by ``Microsimulation(dataset=...)``, e.g.
            ``"USSingleYearDataset"``.
        engine_package: The pip package providing the engine, e.g.
            ``"policyengine-us"`` — named in the install extra and in the
            ImportError when the engine is absent.
    """

    country: str
    year: int
    hf_repo: str
    filename: str
    engine_module: str
    engine_class: str
    engine_package: str
    variant: str = DEFAULT_VARIANT

    @property
    def key(self) -> tuple[str, int, str]:
        """The ``(country, year, variant)`` registry key for this spec."""
        return (self.country, self.year, self.variant)

    @property
    def hf_url(self) -> str:
        """The ``hf://`` URL of the published artifact."""
        return f"hf://{self.hf_repo}/{self.filename}"


#: The published-dataset registry. Keys are ``(country, year, variant)``; the
#: lowest-friction way to publish a new population is one entry here.
REGISTRY: dict[tuple[str, int, str], DatasetSpec] = {}


def register(spec: DatasetSpec) -> DatasetSpec:
    """Add ``spec`` to :data:`REGISTRY`, refusing to silently shadow a key.

    Raises:
        ValueError: If a different spec is already registered for the same
            ``(country, year, variant)`` — re-registering the *same* spec is a
            no-op so re-import is safe.
    """
    existing = REGISTRY.get(spec.key)
    if existing is not None and existing != spec:
        raise ValueError(
            f"A different dataset is already registered for {spec.key}: "
            f"{existing.hf_url} (refusing to shadow it with {spec.hf_url})."
        )
    REGISTRY[spec.key] = spec
    return spec


# ---------------------------------------------------------------------------
# Published datasets. One entry per (country, year). Artifacts live on the Hub.
# ---------------------------------------------------------------------------

register(
    DatasetSpec(
        country="uk",
        year=2023,
        variant=DEFAULT_VARIANT,
        hf_repo="policyengine/populace-uk-private",
        filename="populace_uk_2023.h5",
        engine_module="policyengine_uk.data",
        engine_class="UKSingleYearDataset",
        engine_package="policyengine-uk",
    )
)

register(
    DatasetSpec(
        country="us",
        year=2024,
        variant=DEFAULT_VARIANT,
        hf_repo="policyengine/populace-us",
        filename="populace_us_2024.h5",
        engine_module="policyengine_us.data",
        engine_class="USSingleYearDataset",
        engine_package="policyengine-us",
    )
)
