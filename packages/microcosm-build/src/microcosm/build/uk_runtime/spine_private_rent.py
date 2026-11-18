"""The clone-first spine's private-rent stage, G7 (microcosm#931).

After the SPI housing shell has drawn every SPI-channel household's tenure,
bedrooms and rent, the spine's ``private_rent`` stage
(:class:`UKPrivateRentStageTransform`) reads each Great Britain private
renter's rent off the list of rents of the BRMA its drawn area sits in, as the
dense line's ``uk.full.private_rent`` node did after geography
(microcosm#1262).
:func:`~microcosm.build.uk_runtime.private_rent_localisation.localise_rents`
keeps the renter's rank among the FRS renters of its FRS region and bedroom
size, reads that rank off the list of its BRMA and size, and scales it per FRS
region so that, on the census private-renter mix of the region's BRMAs, the
region's mean rent is the FRS level. On the spine:

- **Each row ranks on its own.** The key ``localise_rents`` groups rows by is
  each row's ``geography_household_key``, unique per row here: the geography
  identity keys every base row and the SPI-copy area stage every SPI-channel
  row. So every geographic clone and every SPI copy is a spine household of
  its own. The dense node keyed the pool clones of one spine household alike
  (``property_localisation.spine_household_keys``). Those keys need the CGT
  lineage, which the spine writes only after this stage, and would give one
  key to the clones of an SPI copy whose housing the shell drew separately
  for each clone. Per-row keys have neither problem (C-14).
- **Tie uniforms.** Each row's tie-break ``v`` is the keyed uniform of
  ``(geography_household_key, "rent_rank")`` on the stage's own stream,
  ``("sha256-u53-v1", "uk-spine-private-rent-v1", 0, seed)``, with the seed 0
  the stage declares (C-43). The rank ``u = F(x-) + v (F(x) - F(x-))`` places
  every row inside its own weight step, tied or not, so two FRS renters at one
  rent in one cell, or two clones of one renter, take ranks of their own.
  Clone 0's key, and so its ``v``, is the same at every clone count; its rank
  and its region's scale pool over clones, so its rent nests approximately.
- **Sources.** ``localise_rents`` counts a region's FRS renters in true source
  households (``receipt_counts``), so a renter's clones and copies count once
  toward the sampled rung's rule, which ranks and scales a region holding
  fewer FRS renters than ``MIN_SAMPLED_REGION_RENTERS`` on Great Britain's.
- **Strictness.** A full-scale spine ranks and scales every region on its own
  FRS renters and refuses a region with none; a sampled rung
  (``sampled_rung``) takes the relaxation.

Only ``rent`` is rewritten: reported Housing Benefit, Universal Credit and
mortgage payments stay as reported (María's ruling of 2026-11-06, item 5),
the weights pass through, and
the stage records its household mass receipt under its own reason. Equality
with the retired dense node is not a goal: it ranked the final frame, where
CGT twins and support copies held keys of their own (C-14).

This module's refusals give columns, dtypes and declared constants only, never
a key, an id, a BRMA or area of a household, a value or a count of households.
It refuses a re-drawn renter whose BRMA has no list before ``localise_rents``
runs, so none of that function's own refusals names a household's BRMA: they
name at most an FRS region, an LHA category, or a BRMA of the census counts
that has no list.

The evidence counts, among the rows of geographic clone 0, where a union over
clones could grow with K, the distinct true source households whose rent is
re-drawn, rounded through ``receipt_counts``, and the (region, category) cells
ranked against their region or against Great Britain; with K, the seed, the
stream, the resources' digests, the method and the strictness (C-44).

``sources.yaml`` declares the stage after ``spi_housing_shell``: the
transform checks the stage it runs as against the reviewed declaration this
module states (name, household grain, no outputs, ``rent`` as its one
rewrite, the reviewed operations verbatim and its two artifacts), which the
declaration carries, its stream and seed among them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime import private_rent_localisation as prl
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.geographic_clones import (
    GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
    geographic_clone_count,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geography_first_roster import PRIVATE_RENT_STAGE
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    rounded_count,
    source_household_ids,
)
from microcosm.frame import Frame, Weights
from microcosm.graph.randomness import keyed_uniform

__all__ = [
    "COUNTS_ARTIFACT_ROLE",
    "COUNTS_RESOURCE",
    "PRIVATE_RENT_OPERATION_KIND",
    "PRIVATE_RENT_STAGE_NAME",
    "QUANTILES_ARTIFACT_ROLE",
    "QUANTILES_RESOURCE",
    "RENT_RANK_PROCESS",
    "UK_PRIVATE_RENT_INPUTS",
    "UK_PRIVATE_RENT_MASS_CONSERVATION_REASON",
    "UK_PRIVATE_RENT_REWRITES",
    "UK_SPINE_PRIVATE_RENT_SEED",
    "UK_SPINE_PRIVATE_RENT_STREAM",
    "UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT",
    "UKPrivateRentResources",
    "UKPrivateRentResult",
    "UKPrivateRentStageTransform",
    "UKSpinePrivateRents",
    "draw_uk_spine_private_rents",
    "load_committed_private_rent_resources",
    "load_private_rent_resources",
    "localise_uk_spine_private_rents",
    "private_rent_operation_parameters",
    "uk_spine_rent_tie_uniforms",
]

PRIVATE_RENT_STAGE_NAME = PRIVATE_RENT_STAGE
PRIVATE_RENT_OPERATION_KIND = "localise_private_rents"
#: The stage's own stream, apart from the retired dense node's, and the seed
#: the stage declares (C-43): one constant, so clone 0's tie-breaks nest at
#: every K.
UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT = "uk-spine-private-rent-v1"
UK_SPINE_PRIVATE_RENT_SEED = 0
UK_SPINE_PRIVATE_RENT_STREAM = (
    "sha256-u53-v1",
    UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT,
    0,
    UK_SPINE_PRIVATE_RENT_SEED,
)
#: The draw's process coordinate after the row's key, as the dense node
#: spelled it.
RENT_RANK_PROCESS = "rent_rank"
#: The stage's two artifacts, whose bytes enter its contract: the BRMAs'
#: weekly lists of rents and the census private-rented households by FRS
#: region, BRMA and bedrooms (the count table ``frs_brma`` reads under the
#: same role).
QUANTILES_ARTIFACT_ROLE = "rent_quantiles"
QUANTILES_RESOURCE = "brma_rent_quantiles.json"
COUNTS_ARTIFACT_ROLE = "count_resource"
COUNTS_RESOURCE = "brma_private_rented_households.json"
_ARTIFACTS = (
    (QUANTILES_ARTIFACT_ROLE, QUANTILES_RESOURCE),
    (COUNTS_ARTIFACT_ROLE, COUNTS_RESOURCE),
)
_RENT_COLUMN = "rent"
_CHANNEL_COLUMN = "household_support_channel"
#: The household columns the stage reads, besides ``source_household_id``,
#: the true source the receipts' reader takes, and the clone index the
#: evidence counts clone 0 by, where the frame carries one.
UK_PRIVATE_RENT_INPUTS = (
    GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
    "region",
    "tenure_type",
    _RENT_COLUMN,
    "num_bedrooms",
    "brma",
    _CHANNEL_COLUMN,
)
UK_PRIVATE_RENT_REWRITES = (_RENT_COLUMN,)
#: The household-mass receipt this stage records (the reviewed
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_PRIVATE_RENT_MASS_CONSERVATION_REASON = (
    "Private rents read off the BRMA lists on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)
_CLONE_INDEX_COLUMN = ladder_clone_index_column("household")
_FLOAT64 = np.dtype("float64")


def private_rent_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operations G7's stage must declare verbatim.

    The rent re-draw: the rows it re-draws (private tenure, a positive rent,
    outside Northern Ireland), the per-row key, tie-break process, stream and
    seed, the cells it ranks in, the channel it ranks against, its two
    resources and the lists' method, the weekly annualisation, the scale band,
    the sampled rung's region rule and the method; then the household mass
    receipt.
    """

    return (
        (
            PRIVATE_RENT_OPERATION_KIND,
            {
                "tenure_type": prl.PRIVATE_TENURE,
                "excluded_regions": [prl.NORTHERN_IRELAND],
                "identity": GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
                "process": RENT_RANK_PROCESS,
                "stream": UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT,
                "seed": UK_SPINE_PRIVATE_RENT_SEED,
                "categories": list(prl.CATEGORIES),
                "reference_channel": prl.FRS_CHANNEL,
                "quantiles_resource": QUANTILES_RESOURCE,
                "quantiles_method": prl.QUANTILES_METHOD,
                "counts_resource": COUNTS_RESOURCE,
                "weeks_in_year": prl.WEEKS_IN_YEAR,
                "scale_band": list(prl.SCALE_BAND),
                "sampled_region_renters": prl.MIN_SAMPLED_REGION_RENTERS,
                "method": prl.METHOD_VERSION,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": UK_PRIVATE_RENT_MASS_CONSERVATION_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


def _assert_private_rent_stage(stage: SourceStageSpec) -> None:
    """The stage the transform runs as: its name, household grain, no outputs
    and ``rent`` as its one rewrite, the reviewed operations verbatim, and
    exactly its two artifacts, each with ``runtime_sha256_required``."""

    if not isinstance(stage, SourceStageSpec):
        raise TypeError("The private-rent stage runs as a SourceStageSpec stage.")
    if stage.stage != PRIVATE_RENT_STAGE_NAME:
        raise ValueError(
            f"Expected stage {PRIVATE_RENT_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The private-rent stage rewrites households; grain must be "
            f"'household', got {stage.grain!r}."
        )
    if tuple(stage.outputs) or tuple(stage.rewrites) != UK_PRIVATE_RENT_REWRITES:
        raise ValueError(
            "The private-rent stage rewrites exactly the rent; the stage must "
            f"declare no outputs and the rewrites {UK_PRIVATE_RENT_REWRITES}, got "
            f"{tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )
    _assert_closed_world_operations(stage, private_rent_operation_parameters())
    declared = sorted(
        (str(artifact.get("role")), str(artifact.get("resource")))
        for artifact in stage.artifacts
        if artifact.get("runtime_sha256_required") is True
    )
    if len(stage.artifacts) != len(_ARTIFACTS) or declared != sorted(_ARTIFACTS):
        raise ValueError(
            f"The private-rent stage reads the artifacts {list(_ARTIFACTS)} (role, "
            "resource), each with runtime_sha256_required: true, and no other; the "
            f"stage declares {len(stage.artifacts)} artifact(s)."
        )


@dataclass(frozen=True)
class UKPrivateRentResources:
    """The lists of rents and the census counts, parsed, with their bytes'
    SHA-256."""

    quantiles: prl.RentQuantiles = field(repr=False)
    counts: Mapping[tuple[str, str], Mapping[str, float]] = field(repr=False)
    quantiles_sha256: str
    counts_sha256: str


def load_private_rent_resources(
    *, quantiles: bytes, counts: bytes
) -> UKPrivateRentResources:
    """The two resources from their bytes, through the localiser's loaders."""

    return UKPrivateRentResources(
        quantiles=prl.load_rent_quantiles(quantiles),
        counts=MappingProxyType(prl.load_private_renter_counts(counts)),
        quantiles_sha256=hashlib.sha256(quantiles).hexdigest(),
        counts_sha256=hashlib.sha256(counts).hexdigest(),
    )


def load_committed_private_rent_resources() -> UKPrivateRentResources:
    """The committed lists of rents and census counts the stage declares."""

    package = files("microcosm.build.uk")
    return load_private_rent_resources(
        quantiles=package.joinpath(QUANTILES_RESOURCE).read_bytes(),
        counts=package.joinpath(COUNTS_RESOURCE).read_bytes(),
    )


def uk_spine_rent_tie_uniforms(keys: Sequence[str]) -> np.ndarray:
    """Each key's tie-break uniform on the stage's stream.

    The keyed uniform of ``(key, "rent_rank")`` on
    :data:`UK_SPINE_PRIVATE_RENT_STREAM`: a function of the key alone, so a
    row's draw does not depend on the other rows, their order or K.
    """

    return keyed_uniform(
        stream=UK_SPINE_PRIVATE_RENT_STREAM,
        keys=[(str(key), RENT_RANK_PROCESS) for key in keys],
    )


def _require_columns(household: pd.DataFrame) -> None:
    missing = [
        column for column in UK_PRIVATE_RENT_INPUTS if column not in household.columns
    ]
    if missing:
        raise ValueError(
            f"The private-rent stage reads household {missing}, which the earlier "
            "spine stages write; the frame lacks them (microcosm#931)."
        )
    dtype = household[_RENT_COLUMN].dtype
    if dtype != _FLOAT64:
        raise ValueError(
            f"The private-rent stage rewrites a float64 {_RENT_COLUMN!r}, bit for "
            f"bit where it keeps it; the column holds {dtype}."
        )


def _redrawn_keys(household: pd.DataFrame, redrawn: np.ndarray) -> list[str]:
    """The re-drawn rows' geography keys, each present and each its own."""

    keys = household[GEOGRAPHY_HOUSEHOLD_KEY_COLUMN].to_numpy(dtype=object)[redrawn]
    if not all(isinstance(key, str) and key for key in keys):
        raise ValueError(
            f"Every re-drawn private renter needs its {GEOGRAPHY_HOUSEHOLD_KEY_COLUMN!r}; "
            "the geography identity and the SPI-copy area stage write it "
            "(microcosm#931)."
        )
    if len(set(keys)) != len(keys):
        raise ValueError(
            "The private-rent stage keys each re-drawn renter's tie-break on its own "
            f"{GEOGRAPHY_HOUSEHOLD_KEY_COLUMN!r}, but re-drawn rows share one: every "
            "SPI-channel row needs the key the SPI-copy area stage gives it "
            "(microcosm#931)."
        )
    return [str(key) for key in keys]


def _require_rent_lists(
    brmas: np.ndarray, categories: np.ndarray, quantiles: prl.RentQuantiles
) -> None:
    """Refuse a re-drawn renter whose BRMA has no list for its category, by a
    text that names no BRMA, before ``localise_rents`` would name it."""

    listed = all(
        isinstance(brma, str) and (brma, str(category)) in quantiles.knots
        for brma, category in zip(brmas, categories, strict=True)
    )
    if not listed:
        raise ValueError(
            "A re-drawn Great Britain private renter's brma has no list of rents "
            f"for its bedroom category in {QUANTILES_RESOURCE}; the BRMA stage "
            "places every Great Britain household in a BRMA the lists cover "
            "(microcosm#931)."
        )


def _clone_zero(household: pd.DataFrame) -> np.ndarray:
    """Rows of geographic clone 0: every row of a table without the index."""

    if _CLONE_INDEX_COLUMN not in household.columns:
        return np.ones(len(household), dtype=bool)
    return household[_CLONE_INDEX_COLUMN].to_numpy(dtype=np.int64) == 0


def _ranked_cells(
    rows: pd.DataFrame, *, counted: np.ndarray, strict: bool
) -> tuple[int, int]:
    """Among the ``counted`` re-drawn rows' (region, category) cells, those
    ranked against their region and those ranked against Great Britain.

    ``rows`` holds every re-drawn row's region, category, channel, weight and
    true source, as ``localise_rents`` holds them. Its routing
    (``private_rent_localisation.rent_uniforms``): a sampled rung ranks every
    cell of a thin region against Great Britain's FRS renters; any other cell
    without an FRS renter of positive weight ranks against its region's.
    """

    thin = prl.sampled_thin_regions(rows, strict=strict)
    reference = rows[(rows["channel"] == prl.FRS_CHANNEL) & (rows["weight"] > 0)]
    held = set(
        zip(reference["region"].tolist(), reference["category"].tolist(), strict=True)
    )
    cells = set(
        zip(
            rows.loc[counted, "region"].tolist(),
            rows.loc[counted, "category"].tolist(),
            strict=True,
        )
    )
    great_britain = sum(1 for region, _ in cells if region in thin)
    own_region = sum(1 for cell in cells if cell[0] not in thin and cell not in held)
    return own_region, great_britain


@dataclass(frozen=True)
class UKSpinePrivateRents:
    """The re-draw on one household table: every row's rent, which rows were
    re-drawn and their tie-break uniforms, and what the evidence counts among
    clone-0 rows. The source count is exact; nothing here prints in the repr
    but the cell counts and the strictness."""

    rent: pd.Series = field(repr=False)
    redrawn: np.ndarray = field(repr=False)
    tie_uniforms: pd.Series = field(repr=False)
    source_households_redrawn: int = field(repr=False)
    cells_ranked_against_their_region: int
    cells_ranked_against_great_britain: int
    strict: bool


def draw_uk_spine_private_rents(
    household: pd.DataFrame,
    weights: Sequence[float] | np.ndarray,
    *,
    resources: UKPrivateRentResources,
    strict: bool,
) -> UKSpinePrivateRents:
    """Every household row's rent, Great Britain private renters' re-drawn.

    ``localise_rents`` with each row as its own spine household (its
    ``geography_household_key``), its true source household, the tie-break
    uniforms :func:`uk_spine_rent_tie_uniforms` gives the re-drawn rows'
    keys, and ``strict``. ``weights`` are the rows' household weights, in row
    order. Returns the rents on ``household``'s index in its row order, every
    row ``localise_rents`` does not re-draw bit for bit. Each row's draw
    depends on its own key and the cells' pooled ranks and scales, never on
    the table's row order.

    Refused, by name: a missing input column, a rent column that is not
    float64, weights that do not align, a re-drawn row without its own key,
    a re-drawn row whose BRMA has no list for its category, and every refusal
    of ``localise_rents`` and of the receipts' source reader.
    """

    _require_columns(household)
    weight = np.asarray(weights, dtype=float)
    if weight.shape != (len(household),):
        raise ValueError(
            "The private-rent stage reads one household weight per household row."
        )
    sources = source_household_ids(household, stage="The private-rent stage")
    region = household["region"].to_numpy(dtype=object)
    tenure = household["tenure_type"].to_numpy(dtype=object)
    rent = household[_RENT_COLUMN].to_numpy(dtype=float)
    redrawn = prl.redraw_mask(region, tenure, rent)
    keys = _redrawn_keys(household, redrawn)
    bedrooms = household["num_bedrooms"].to_numpy()
    categories = prl.rent_categories(bedrooms[redrawn])
    brma = household["brma"].to_numpy(dtype=object)
    _require_rent_lists(brma[redrawn], categories, resources.quantiles)
    uniforms = uk_spine_rent_tie_uniforms(keys)
    spine_key = np.full(len(household), None, dtype=object)
    spine_key[redrawn] = keys
    channel = household[_CHANNEL_COLUMN].to_numpy(dtype=object)
    new_rent, _ = prl.localise_rents(
        spine_key=spine_key,
        source=sources,
        region=region,
        tenure_type=tenure,
        rent=rent,
        num_bedrooms=bedrooms,
        brma=brma,
        channel=channel,
        weight=weight,
        tie_uniform=dict(zip(keys, (float(v) for v in uniforms), strict=True)),
        quantiles=resources.quantiles,
        counts=resources.counts,
        strict=bool(strict),
    )
    counted = _clone_zero(household)[redrawn]
    rows = pd.DataFrame(
        {
            "region": region[redrawn].astype(str),
            "category": categories,
            "channel": channel[redrawn].astype(str),
            "weight": weight[redrawn],
            "source": sources[redrawn],
        }
    )
    own_region, great_britain = _ranked_cells(rows, counted=counted, strict=strict)
    return UKSpinePrivateRents(
        rent=pd.Series(new_rent, index=household.index, name=_RENT_COLUMN),
        redrawn=redrawn,
        tie_uniforms=pd.Series(uniforms, index=household.index[redrawn], dtype=float),
        source_households_redrawn=distinct_carriers(sources[redrawn][counted].tolist()),
        cells_ranked_against_their_region=own_region,
        cells_ranked_against_great_britain=great_britain,
        strict=bool(strict),
    )


@dataclass(frozen=True)
class UKPrivateRentResult:
    """G7's frame and what its evidence reads of it, counted among clone-0
    rows; the source count is exact and prints only rounded."""

    frame: Frame = field(repr=False)
    clone_count: int
    strict: bool
    quantiles_sha256: str
    counts_sha256: str
    source_households_redrawn: int = field(repr=False)
    cells_ranked_against_their_region: int
    cells_ranked_against_great_britain: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": PRIVATE_RENT_STAGE_NAME,
            "clone_count": int(self.clone_count),
            "counted_among": "clone_0",
            "source_households_redrawn": rounded_count(self.source_households_redrawn),
            "cells_ranked_against_their_region": int(
                self.cells_ranked_against_their_region
            ),
            "cells_ranked_against_great_britain": int(
                self.cells_ranked_against_great_britain
            ),
            "strict": bool(self.strict),
            "seed": UK_SPINE_PRIVATE_RENT_SEED,
            "stream": list(UK_SPINE_PRIVATE_RENT_STREAM),
            "quantiles_sha256": self.quantiles_sha256,
            "counts_sha256": self.counts_sha256,
            "method": prl.METHOD_VERSION,
            "weights": "unchanged",
            "rewrites": list(UK_PRIVATE_RENT_REWRITES),
        }


def localise_uk_spine_private_rents(
    frame: Frame,
    *,
    resources: UKPrivateRentResources,
    sampled_rung: bool = False,
) -> UKPrivateRentResult:
    """``frame`` with every Great Britain private renter's rent re-drawn (G7).

    :func:`draw_uk_spine_private_rents` on the frame's households and
    weights, strict unless ``sampled_rung``. Rows, ids, weights, every other
    column and the metadata are the frame's own; the mass log gains the
    stage's receipt. Refused, by name: what that function refuses.
    """

    if not isinstance(frame, Frame):
        raise TypeError("The private-rent stage rewrites the households of a Frame.")
    if not isinstance(sampled_rung, bool):
        raise TypeError("sampled_rung says whether the spine is sampled: a bool.")
    validate_uk_national_frame(frame)
    household = frame.table("household")
    # The household weights pass through unchanged. Annotated so the stage's
    # static walk meets microcosm.frame's Weights module.
    weights: Weights = frame.weights_for("household")
    draw = draw_uk_spine_private_rents(
        household, weights.values, resources=resources, strict=not sampled_rung
    )
    rewritten = household.copy()
    rewritten[_RENT_COLUMN] = draw.rent
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = rewritten
    result = Frame(
        tables,
        frame.schema,
        {"household": weights},
        frame.strata,
        mass_log=(
            *frame.mass_log,
            uk_household_mass_conservation_receipt(
                frame, UK_PRIVATE_RENT_MASS_CONSERVATION_REASON
            ),
        ),
        metadata=frame.metadata,
    )
    validate_uk_national_frame(result)
    return UKPrivateRentResult(
        frame=result,
        clone_count=geographic_clone_count(household),
        strict=draw.strict,
        quantiles_sha256=resources.quantiles_sha256,
        counts_sha256=resources.counts_sha256,
        source_households_redrawn=draw.source_households_redrawn,
        cells_ranked_against_their_region=draw.cells_ranked_against_their_region,
        cells_ranked_against_great_britain=draw.cells_ranked_against_great_britain,
    )


@dataclass(frozen=True)
class UKPrivateRentStageTransform:
    """Whole-stage transform for the private rents (G7).

    ``sampled_rung`` says whether the spine is sampled: a full-scale spine is
    strict. Constructing the transform checks the stage it runs as and loads
    the committed lists of rents and census counts; running it calls
    :func:`localise_uk_spine_private_rents`.
    """

    stage: SourceStageSpec
    sampled_rung: bool = False
    resources: UKPrivateRentResources = field(init=False, repr=False, compare=False)
    last_result: UKPrivateRentResult | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        _assert_private_rent_stage(self.stage)
        if not isinstance(self.sampled_rung, bool):
            raise TypeError("sampled_rung says whether the spine is sampled: a bool.")
        object.__setattr__(self, "resources", load_committed_private_rent_resources())

    def __call__(self, frame: Frame) -> Frame:
        _assert_private_rent_stage(self.stage)
        result = localise_uk_spine_private_rents(
            frame, resources=self.resources, sampled_rung=self.sampled_rung
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
