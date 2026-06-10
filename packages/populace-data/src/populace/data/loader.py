"""Resolve, download, and load a published populace population.

Three functions over the :mod:`populace.data.registry`: :func:`available` lists
what is published, :func:`download` fetches the artifact from the Hub (cached),
and :func:`load` returns it as the country engine's dataset object — the thing
``Microsimulation(dataset=...)`` consumes. The engine is imported lazily and per
spec, so the base install stays engine-free and a missing engine yields a named
error pointing at the right extra.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from populace.data.registry import REGISTRY, DatasetSpec

__all__ = ["available", "resolve", "latest_year", "download", "load"]


def available() -> list[tuple[str, int]]:
    """Every published ``(country, year)``, sorted."""
    return sorted(REGISTRY)


def latest_year(country: str) -> int:
    """The most recent published year for ``country``.

    Raises:
        ValueError: If no dataset is published for ``country``.
    """
    country = country.lower()
    years = [yr for (c, yr) in REGISTRY if c == country]
    if not years:
        published = sorted({c for c, _ in REGISTRY})
        raise ValueError(
            f"No populace dataset for country {country!r}; published "
            f"countries: {published}."
        )
    return max(years)


def resolve(country: str, year: int | None = None) -> DatasetSpec:
    """Look up the :class:`DatasetSpec` for ``country`` (and ``year``).

    Args:
        country: Country slug (case-insensitive), e.g. ``"us"``.
        year: Population year; ``None`` selects the latest published year for
            the country.

    Raises:
        ValueError: If the country (or the country/year pair) is not published,
            naming what *is* available.
    """
    country = country.lower()
    if year is None:
        year = latest_year(country)
    spec = REGISTRY.get((country, int(year)))
    if spec is None:
        years = sorted(yr for c, yr in REGISTRY if c == country)
        if years:
            detail = f"published years for {country!r}: {years}"
        else:
            published = sorted({c for c, _ in REGISTRY})
            detail = f"no datasets for {country!r}; published countries: {published}"
        raise ValueError(f"No populace dataset for ({country!r}, {year}); {detail}.")
    return spec


def download(country: str, year: int | None = None) -> Path:
    """Download (and cache) the published artifact, returning its local path.

    Uses ``huggingface_hub``'s cache, so repeated calls do not re-download.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        ValueError: If the country/year is not published.
    """
    spec = resolve(country, year)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "populace-data needs huggingface_hub to download artifacts; "
            "reinstall populace-data with its dependencies."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            repo_type="dataset",
        )
    )


def load(country: str, year: int | None = None):
    """Return a published population as its engine dataset object.

    The returned object is accepted directly by the country's
    ``Microsimulation(dataset=...)`` — e.g. for the US::

        from policyengine_us import Microsimulation
        from populace.data import load

        sim = Microsimulation(dataset=load("us", 2024))

    Args:
        country: Country slug (case-insensitive), e.g. ``"us"``.
        year: Population year; ``None`` loads the latest published year.

    Raises:
        ImportError: If the country engine (e.g. ``policyengine-us``) is not
            installed — the message names the ``populace-data[<country>]`` extra.
        ValueError: If the country/year is not published.
    """
    spec = resolve(country, year)
    try:
        module = importlib.import_module(spec.engine_module)
    except ImportError as exc:
        raise ImportError(
            f"populace-data needs {spec.engine_package} to load the {spec.country!r} "
            f"population; install it with: pip install 'populace-data[{spec.country}]'."
        ) from exc
    dataset_class = getattr(module, spec.engine_class)
    return dataset_class(file_path=str(download(country, year)))
