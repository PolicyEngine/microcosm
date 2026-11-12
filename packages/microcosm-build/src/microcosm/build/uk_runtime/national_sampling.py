"""Scale-ladder sampling for the UK spine (#627).

A spine built below f100 samples its FRS root household by household
(:func:`sample_uk_spine_root`) through :class:`UKSampledSpineRootTransform`,
which the root node hashes with this module, and below f100 with the shared
:mod:`microcosm.build.frame_sampling` core too. Sampled mass is renormalized
to the root's household total (the US stacked semantics #627 names). The
module also holds the rung tokens and the default sampling seed the UK builds
share.

The dense line samples nothing of its own: it takes the bound spine's
sampling with its clones and areas (microcosm#931), so its sampler, which drew
whole source families from a spine, is retired.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping

from microcosm.build import frame_sampling
from microcosm.build.frame_sampling import (
    normalize_sampled_household_mass,
    sample_frame_households,
)
from microcosm.frame import Frame

__all__ = [
    "UK_SAMPLE_RUNG_TOKENS",
    "UK_SAMPLE_SEED_DEFAULT",
    "UKSampledSpineRootTransform",
    "sample_uk_spine_root",
]

#: The #624 scale-ladder rungs and their identity tokens, shared with the US
#: stacked pipeline (``LOGBOOK_RUNGS``): ~1% smoke, 10% dev, full scale.
UK_SAMPLE_RUNG_TOKENS: Mapping[float, str] = {
    0.01: "f001",
    0.10: "f010",
    1.00: "f100",
}

#: Default survey-sampling seed, matching the US stacked pipeline's. Distinct
#: from the build ``--seed`` so dev-scale seed sweeps vary one draw at a time.
UK_SAMPLE_SEED_DEFAULT = 578


def sample_uk_spine_root(
    frame: Frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Frame, dict[str, object] | None]:
    """Sample the spine build's FRS root at a rung below f100.

    Households are drawn by the shared core and their mass is renormalized
    to the root's household total. At f100 the frame is returned as it is,
    with no receipt.
    """

    if fraction == 1.0:
        return frame, None
    household_weight = frame.weights_for("household")
    pre_households = int(len(frame.table("household")))
    sampled, receipt = sample_frame_households(
        frame,
        fraction=fraction,
        seed=seed,
        source_name="UK FRS spine",
    )
    normalized, factor = normalize_sampled_household_mass(
        sampled,
        target_mass=float(household_weight.total),
        source_name="UK FRS spine",
    )
    return normalized, {
        "fraction": float(fraction),
        "seed": int(seed),
        "rung_token": UK_SAMPLE_RUNG_TOKENS[fraction],
        "pre_household_count": pre_households,
        "post_household_count": int(len(normalized.table("household"))),
        "normalization_factor": float(factor),
        "receipt": dict(receipt),
    }


class UKSampledSpineRootTransform:
    """CREATE-stage adapter applying the declared sampling rung at ingest.

    The spine build wraps the FRS root's transform in it. Its sampling code
    lives here and in the shared core, not in the root stage's module, so it
    declares them for the root node to hash (microcosm#931): this module at
    every rung, since it resets the root's indexes at f100 too, and the shared
    core only below f100, since :func:`sample_uk_spine_root` returns before
    calling it at f100.
    """

    def __init__(
        self,
        transform,
        *,
        fraction: float,
        seed: int,
    ) -> None:
        self.transform = transform
        self.fraction = fraction
        self.seed = seed
        self.sampling: dict[str, object] | None = None

    def graph_implementation_dependencies(self) -> tuple[object, ...]:
        """The modules the root node hashes for this adapter's sampling."""

        if self.fraction == 1.0:
            return (sys.modules[__name__],)
        return (sys.modules[__name__], frame_sampling)

    def _sample(self, assembled):
        sampled, self.sampling = sample_uk_spine_root(
            assembled,
            fraction=self.fraction,
            seed=self.seed,
        )
        # Graph populations use row positions as their internal alignment
        # index. Frame sampling preserves source DataFrame indexes by design,
        # so normalize those indexes at this adapter boundary.
        tables = {
            entity: sampled.table(entity).reset_index(drop=True)
            for entity in sampled.entities
        }
        tables.update(
            {name: sampled.link(name).reset_index(drop=True) for name in sampled.links}
        )
        return Frame(
            tables,
            sampled.schema,
            {
                entity: sampled.weights_for(entity)
                for entity in sampled.weighted_entities
            },
            sampled.strata.reset_index(drop=True),
            mass_log=sampled.mass_log,
            metadata=sampled.metadata,
        )

    def effective_fraction(self) -> float:
        """Return the configured input sampling fraction."""

        return float(self.fraction)

    def __call__(self, frame):
        return self._sample(self.transform(frame))

    def run_with_sources(self, frame, sources):
        runner = getattr(self.transform, "run_with_sources", None)
        assembled = (
            runner(frame, sources) if callable(runner) else self.transform(frame)
        )
        return self._sample(assembled)

    def checkpoint_metadata(self) -> dict[str, object]:
        hook = getattr(self.transform, "checkpoint_metadata", None)
        if not callable(hook):
            raise RuntimeError("FRS root transform exposes no checkpoint metadata.")
        return dict(hook())
