"""The clone-first UK spine at four geographic clones on the H2 fixture.

microcosm#931 groups (b) and (c). The H2 fixture's invented sources run the
whole spine in this process twice, through the real engine: at one geographic
clone, the fixture's own config, and at four, the same config with
``geographic_clones`` 4, so the graph's geographic_expand node, the transform
that makes the clones and the engine all take K = 4. Each run records, per
graph node, the simulations each engine call ran, the population's mass, its
clone-index invariant and each stage's outputs as weighted shares, and keeps
the populations the exact checks compare.

Too slow for every pull request, the module runs after merge only, in the
engine-uk-after-merge job on pushes to main.

- **Every stage runs**, and at K = 4 every engine call from geographic_expand
  on runs four simulations, one per clone, in as many calls per stage as at
  one clone: no stage calls the engine per clone. private_school_attendance
  restates attendance over the pool, the batched route.
- **The clone-index invariant** holds at every node's population from
  geographic_expand on.
- **Exact:** clone 0 after geographic_expand and geographic_identity is the
  one-clone frame on every column, at K x its weight bitwise. Through
  area_brma it is still the one-clone frame on every column: the area draw
  (atomic_area_assign) keys each row's draw on its geography key, which
  clone 0 shares with one clone, and picks its fallback level from its
  source's own inputs, the local authority and the fuel-relief flag read the
  drawn area, frs_take_up's anchored residual and frs_person_draws'
  apprentice walk count within each clone, the walk on prefixes that 1/4
  scales exactly, every other draw is keyed on the row's own id, and
  area_brma reads each Great Britain household's BRMA off its drawn area (the
  fixture holds no Northern Ireland household, whose BRMA would read the
  LHA_category the engine computes for clone 0 from that clone's rows).
  spi_support_channel copies K = 1's households in clone 0, at nesting
  weights, spi_income_band_donors seats K = 1's households in K = 1's bands
  with K = 1's carriers, and spi_copy_areas gives clone 0's copies K = 1's
  keys and areas (microcosm#931).
- **The band donors' weights:** their mass scale is reported first. The stage
  funds every clone's donors at one scale, the smallest any clone's region
  allows, and on this fixture's capped rung that scale at K = 4 is below one
  clone's. So clone 0's donors weigh K = 1's times the ratio of the two
  scales, and each funding stratum's incumbents K = 1's times one ratio of
  the stratum's own: K x clone-0 weights nest on one clone's only where the
  two scales are equal. Whether the stage takes a scale per clone is open
  (microcosm#931).
- **Tolerance classes:** the draws keep their K-invariant receipt fields, and
  the area stages and private_rent every receipt field but K; from
  frs_take_up on, every output every clone carries, at a weighted share near
  one clone's (private_rent's ranks and scales pool over the clones, so its
  rents nest only so, and so do the main residences the WAS chain's area home
  values read off each owner's constituency at a rank within its region, G8,
  and the regional uprating that divides by their kept draw), and every node
  keeps K = 1's household mass.
- **Receipts:** the clone count in geographic_expand's, spi_support_channel's
  and spi_income_band_donors' receipts and, at K = 4 only, the support clips'
  and property components' (was_wealth's area home values block carries its
  own at every K); the geographic fold receipts at the six folding fits at
  K = 4, and none at one clone.
- **Areas and keys, at one clone and at four:** from area_brma on every
  household carries its area, ladder, local authority, fuel-relief flag and
  BRMA; the geography keys are unique through area_brma, the SPI copies and
  band donors repeat the key they copy until spi_copy_areas gives each its
  own, and every later copy carries a key spi_copy_areas gave; private_rent
  re-draws Great Britain's private renters' rents and leaves every other
  row's, and H2's root households all own their homes, so it re-draws none
  here (microcosm#931).
- **Home values, at one clone and at four:** after was_wealth every England
  and Wales owner with a positive draw holds the value the area home values
  read off its constituency's knots, with the draw beside it, and every other
  row the draw bit for bit; the draw reaches the spine's last population and
  the release boundary drops it. was_wealth's values are the area home values
  on the frame the stage ran on, bit for bit; regional_property_uprating
  returns each localised home to its knot within one rounding, and a home at
  the knots' cap to the cap or a few roundings below; and every high-value
  support copy keeps its parent's every column but its ids (microcosm#931).

It cannot catch within-batch int32 collisions: ids past 2**31 reach the engine
at K = 1 already, and these households are too few. The adapter's rank rule
handles them (microcosm.frame.adapters.engine_identifiers): an entity whose
ids would repeat once stored takes 1 plus each id's rank within its batch,
and microcosm-frame's test_engine_identifiers.py and
test_policyengine_uk_identifiers.py pin that on invented frames, so a
licensed build's per-batch int32 count is no longer the only check. Invented
data only;
assertion messages give K, stages, columns, counts and booleans, never a
weight, a value or an id.
"""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import copy_id_table, spi_band_donors, spi_support
from microcosm.build.uk_runtime import private_rent_localisation as prl
from microcosm.build.uk_runtime.area_home_values import (
    AREA_HOME_VALUES_STREAM,
    FRS_REGION_TO_KNOTS_REGION,
    load_committed_area_home_values_resources,
    localise_area_home_values,
)
from microcosm.build.uk_runtime.geographic_clones import (
    GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geographic_expand import (
    GEOGRAPHIC_EXPAND_STAGE_NAME,
    UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS,
)
from microcosm.build.uk_runtime.geographic_identity import (
    GEOGRAPHIC_IDENTITY_STAGE_NAME,
)
from microcosm.build.uk_runtime.graph import uk_registry, uk_spine_graph
from microcosm.build.uk_runtime.graph_evidence import load_spine_stage_artifacts
from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
from microcosm.build.uk_runtime.high_value_home_support import (
    HIGH_VALUE_SUPPORT_COPIES_COLUMN,
    HIGH_VALUE_SUPPORT_COPY_ID_FIELD,
    HIGH_VALUE_SUPPORT_STAGE_NAME,
)
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_FOLD_EVIDENCE,
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_release_export_frame,
)
from microcosm.build.uk_runtime.private_school_attendance import (
    PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE,
    PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE,
)
from microcosm.build.uk_runtime.property_localisation import (
    RANK_BASIS_COLUMN,
    localise_main_residence_values,
    owner_mask,
    owner_uniforms,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    rounded_count,
    source_household_ids,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR,
    PERSON_IS_SPI_INCOME_BAND_CARRIER,
    SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN,
    UK_SPI_INCOME_BAND_DONORS_STAGE_NAME,
)
from microcosm.build.uk_runtime.spi_support import (
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    UK_SPI_SUPPORT_STAGE_NAME,
    support_channel_column,
    support_source_id_column,
)
from microcosm.build.uk_runtime.spine_geography import (
    AREA_BRMA_STAGE_NAME,
    ATOMIC_AREA_ASSIGN_STAGE_NAME,
    LOCAL_AUTHORITY_STAGE_NAME,
    RFDR_STAGE_NAME,
    SPI_COPY_AREAS_STAGE_NAME,
    UK_SPI_COPY_AREAS_REWRITE_COLUMNS,
)
from microcosm.build.uk_runtime.spine_private_rent import PRIVATE_RENT_STAGE_NAME
from microcosm.build.uk_runtime.was_wealth import (
    AREA_HOME_VALUES_EVIDENCE,
    UK_WAS_WEALTH_STAGE_NAME,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_graph.acceptance_h_parity import UK_SPINE_PARITY, _require

#: The clone count held to the fixture's one clone.
_CLONES = 4
G1 = GEOGRAPHIC_EXPAND_STAGE_NAME
G1B = GEOGRAPHIC_IDENTITY_STAGE_NAME
SPI = UK_SPI_SUPPORT_STAGE_NAME
DONORS = UK_SPI_INCOME_BAND_DONORS_STAGE_NAME
COPY_AREAS = SPI_COPY_AREAS_STAGE_NAME
#: The spine's stage count since the area stages (microcosm#931): 54, G1 and
#: G1b, then atomic_area_assign, local_authority, rfdr, spi_copy_areas and
#: private_rent, with area_brma in frs_brma's place, less regional_home_values,
#: whose England and Wales home values the WAS chain now reads.
_STAGE_COUNT = 60
#: The area stages before the draws, which key each row's area on its
#: geography key, as G1b left it.
_AREA_STAGES = (
    ATOMIC_AREA_ASSIGN_STAGE_NAME,
    LOCAL_AUTHORITY_STAGE_NAME,
    RFDR_STAGE_NAME,
)
#: The first stage after the area stages, and the last before
#: spi_support_channel, which is the first engine call after G1.
_FIRST_DRAW = "frs_take_up"
_LAST_DRAW = AREA_BRMA_STAGE_NAME
#: The draws that count within each geographic clone before
#: spi_support_channel: frs_take_up's anchored residual takes
#: ``int(rate x units)`` of each clone's units, and frs_person_draws'
#: apprentice walk meets the rate times each clone's weight aged 21 to 24, so
#: clone 0's draws are one clone's (microcosm#931). Every other draw from
#: frs_take_up to area_brma is keyed on the row's own id, which clone 0
#: shares with one clone.
_COUNTED_WITHIN_EACH_CLONE = {
    ("benunit", "would_claim_child_benefit"),
    ("benunit", "would_claim_pc"),
    ("benunit", "would_claim_uc"),
    ("person", "is_apprentice"),
}
#: area_brma's BRMA: Great Britain's off the drawn area, Northern Ireland's
#: from a count-table cell that reads the engine's LHA_category.
_BRMA = ("household", "brma")
#: The stages whose every receipt field but K is one clone's: the area
#: stages count their sources over every clone where each is the same at
#: every K, and among clone-0 rows where a union could grow with K
#: (microcosm#931).
_K_INVARIANT_RECEIPTS = (*_AREA_STAGES, _LAST_DRAW, COPY_AREAS, PRIVATE_RENT_STAGE_NAME)
#: The housing shell, whose rents private_rent reads.
_HOUSING = "spi_housing_shell"
#: The WAS stage, whose chain reads England and Wales home values off each
#: owner's constituency (the area home values, G8; microcosm#931).
_WAS = UK_WAS_WEALTH_STAGE_NAME
#: The regional property uprating, which divides by the draw the WAS stage
#: keeps, and the high-value support, whose copies keep their parents'
#: values and areas (microcosm#931).
_UPRATING = "regional_property_uprating"
_HIGH_VALUE = HIGH_VALUE_SUPPORT_STAGE_NAME
#: The populations the exact checks and the area and key checks read, by
#: stage.
_KEPT = (
    G1,
    G1B,
    _LAST_DRAW,
    SPI,
    DONORS,
    COPY_AREAS,
    _HOUSING,
    PRIVATE_RENT_STAGE_NAME,
    _WAS,
    _UPRATING,
    _HIGH_VALUE,
)
#: The label of the spine's last population, which the key checks read too.
_FINAL = "the last node"
#: The fits that fold geographic clones into their survey records, each
#: receipting the fold at K > 1 (hmrc_spi_income_spine's two forests share one
#: receipt).
_FOLDING_STAGES = frozenset(
    {
        "hmrc_spi_income_spine",
        "spi_employment_shell",
        "spi_housing_shell",
        "uc_reporter_redraw",
        "uc_capital_coherence",
        "salary_sacrifice",
    }
)
#: The six support-clip stages and property components, whose receipts carry
#: the clone count their gates scale by only at K > 1.
_CLONE_SCALED_RECEIPTS = frozenset(
    {
        "was_wealth",
        "was_lisa",
        "nts_bus_travel",
        "lcfs_consumption",
        "etb_vat",
        "etb_services",
        "property_components",
    }
)
#: Each stage's exact-total correction, whose inputs the runs record so the
#: rows it moved are where the stage's output weights differ from them.
_CORRECTIONS = {
    SPI: (spi_support, "_importance_weights_with_exact_total"),
    DONORS: (spi_band_donors, "_exact_incumbent_total"),
}
#: Machine epsilon, the 2.2e-16 of the corrected rows' tolerance.
_EPSILON = float(np.finfo(np.float64).eps)
#: How far a share over every clone may sit from one clone's. The clones draw
#: on their own keyed uniforms, and from spi_income_band_donors on at its
#: common mass scale, so a weighted share over all four differs from one
#: clone's by the draws' sampling error on the fixture's 135 households, which
#: stays under 0.15 on every output here. The bound still catches a stage that
#: leaves three clones of four unset wherever one clone's share passes 0.27,
#: and the presence check catches it at any share.
_SHARE_TOLERANCE = 0.2
#: Below this many rows a value of one clone's draws can be missing from
#: another clone's own draws by chance, so the presence check reads only
#: outputs that one clone gives at least this many rows.
_PRESENCE_ROWS = 3
#: The relative tolerance of a float a stage sums in another order at K: a
#: household mass, or a receipt target over the clones' rows.
_SUM_RTOL = 1e-12


@dataclass(frozen=True)
class _Run:
    """One clone count's run of the whole spine, as each node left it."""

    clones: int
    order: tuple[str, ...]
    stage_names: tuple[str, ...]
    hits: Mapping[str, bool]
    calls: Mapping[str, tuple[int, ...]]
    masses: Mapping[str, float]
    problems: Mapping[str, tuple[str, ...]]
    shares: Mapping[tuple[str, str], tuple[float, tuple[int, ...]]]
    frames: Mapping[str, Frame]
    handed: Mapping[str, tuple[np.ndarray, ...]]
    evidence: Mapping[str, Mapping[str, object]]


def _sources(directory: Path, clones: int) -> Path:
    """The H2 fixture's sources, at ``clones`` geographic clones."""

    source = _require(UK_SPINE_PARITY / "sources", "tools/graph_uk_spine_fixture.py")
    if clones == 1:
        return source
    target = directory / "sources"
    shutil.copytree(source, target)
    descriptor = json.loads((target / "fixture.json").read_text(encoding="utf-8"))
    assert descriptor["config"]["geographic_clones"] == 1
    descriptor["config"]["geographic_clones"] = clones
    (target / "fixture.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _output_node(nodes: set[str], stage: str) -> str:
    """The node whose population holds ``stage``'s outputs."""

    if stage == "frs_spine":
        return "create_uk_frs"
    return f"{stage}.owned" if f"{stage}.owned" in nodes else stage


def _entity_weights(frame: Frame, entity: str) -> np.ndarray:
    """Each row's household weight: a person's household's, a benefit unit's
    members' household's."""

    household = frame.table("household")
    weights = pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=household["household_id"].to_numpy(),
    )
    if entity == "household":
        return weights.to_numpy()
    person = frame.table("person")
    homes = person["person_household_id"]
    if entity == "person":
        return weights.loc[homes.to_numpy()].to_numpy()
    unit_homes = homes.groupby(person["person_benunit_id"].to_numpy()).first()
    units = frame.table("benunit")["benunit_id"].to_numpy()
    return weights.loc[unit_homes.loc[units].to_numpy()].to_numpy()


def _nonzero(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        return (numbers != 0.0) & ~np.isnan(numbers)
    return values.fillna("").astype(str).ne("").to_numpy()


def _clone_index_problems(frame: Frame, clones: int) -> tuple[str, ...]:
    """Where ``frame`` breaks the clone-index invariant: a table without its
    index, an index of another dtype or other values than exactly 0 to
    K - 1, and members whose index is not their household's and benefit
    unit's."""

    problems = []
    for entity in UK_NATIONAL_SCHEMA.entities:
        column = ladder_clone_index_column(entity)
        table = frame.table(entity)
        if column not in table.columns:
            problems.append(f"the {entity} table lacks {column!r}")
            continue
        if table[column].dtype != np.dtype("int64"):
            problems.append(f"{column!r} is not int64")
        if sorted(pd.unique(table[column]).tolist()) != list(range(clones)):
            problems.append(f"{column!r} does not take exactly the values 0 to K - 1")
    if problems:
        return tuple(problems)
    person = frame.table("person")
    own = person[ladder_clone_index_column("person")].to_numpy()
    for group in UK_NATIONAL_SCHEMA.group_entities:
        units = frame.table(group).set_index(UK_NATIONAL_SCHEMA.id_column(group))
        theirs = person[UK_NATIONAL_SCHEMA.membership_column(group)].map(
            units[ladder_clone_index_column(group)]
        )
        off = int((theirs.to_numpy() != own).sum())
        if off:
            problems.append(
                f"{off} person row(s) carry another index than their {group}"
            )
    return tuple(problems)


def _run(directory: Path, clones: int, pending: Mapping[str, list]) -> _Run:
    source = _sources(directory, clones)
    stages, implementations = fixture_stage_plan_inputs(source)
    stage_names = tuple(stage.stage for stage in stages)
    by_name = {stage.stage: stage for stage in stages}
    graph = uk_spine_graph(geographic_clones=clones)
    compiled = compile_graph(graph)
    order = tuple(compiled.order)
    nodes = set(order)
    first_clone_node = order.index(G1)
    outputs = {
        _output_node(nodes, stage): stage
        for stage in stage_names[stage_names.index(_FIRST_DRAW) :]
    }
    kept = {_output_node(nodes, stage): stage for stage in _KEPT}
    assert order[-1] not in kept
    kept[order[-1]] = _FINAL
    calls: dict[str, tuple[int, ...]] = {}
    masses: dict[str, float] = {}
    problems: dict[str, tuple[str, ...]] = {}
    shares: dict[tuple[str, str], tuple[float, tuple[int, ...]]] = {}
    frames: dict[str, Frame] = {}
    handed: dict[str, tuple[np.ndarray, ...]] = {}

    def observe(node_id: str, population) -> None:
        frame = population.frame
        calls[node_id] = tuple(pending["calls"])
        pending["calls"].clear()
        for stage in _CORRECTIONS:
            if pending[stage]:
                handed[node_id] = tuple(pending[stage])
                pending[stage].clear()
        masses[node_id] = math.fsum(frame.weights_for("household").values)
        if order.index(node_id) >= first_clone_node:
            problems[node_id] = _clone_index_problems(frame, clones)
        else:
            problems[node_id] = tuple(
                f"the {entity} table carries its clone index before {G1}"
                for entity in frame.entities
                if ladder_clone_index_column(entity) in frame.table(entity).columns
            )
        if node_id in outputs:
            stage = by_name[outputs[node_id]]
            for column in (*stage.outputs, *stage.rewrites):
                holders = [
                    entity
                    for entity in frame.entities
                    if column in frame.table(entity).columns
                ]
                if not holders:
                    # Recorded so the share test names it; no stage drops
                    # a column it declares.
                    shares[(stage.stage, column)] = (math.nan, ())
                    continue
                (entity,) = holders
                table = frame.table(entity)
                present = _nonzero(table[column])
                weights = _entity_weights(frame, entity)
                index = table[ladder_clone_index_column(entity)].to_numpy()
                shares[(stage.stage, column)] = (
                    float(weights[present].sum() / weights.sum()),
                    tuple(
                        int(present[index == clone].sum()) for clone in range(clones)
                    ),
                )
        if node_id in kept:
            frames[kept[node_id]] = frame

    store = ContentStore(directory / "store")
    manifest = run_graph(
        compiled,
        sources={"frs": source},
        store=store,
        # Built against the graph the run is handed, as the spine build does.
        kernels=uk_registry(dict(implementations), graph=graph),
        resume="forbid",
        decisions=(),
        retain_populations=(compiled.versions[order[-1]],),
        _population_observer=observe,
    )
    artifacts = load_spine_stage_artifacts(manifest, store, stage_names=stage_names)
    return _Run(
        clones=clones,
        order=order,
        stage_names=stage_names,
        hits={node: receipt.hit for node, receipt in manifest.nodes.items()},
        calls=calls,
        masses=masses,
        problems=problems,
        shares=shares,
        frames=frames,
        handed=handed,
        evidence={stage: document["evidence"] for stage, document in artifacts.items()},
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[int, _Run]:
    """The H2 fixture's whole spine at one clone and at four, each engine
    call's simulations and each exact-total correction's inputs recorded."""

    pending: dict[str, list] = {"calls": [], **{stage: [] for stage in _CORRECTIONS}}
    materialize = PolicyEngineUKEngine.materialize

    def counted(self, bundle, variables, period, **kwargs):
        pending["calls"].append(self.simulation_batches(bundle))
        return materialize(self, bundle, variables, period, **kwargs)

    def recording(function, stage: str):
        def recorded(values, *args, **kwargs):
            pending[stage].append(np.array(values, dtype=float, copy=True))
            return function(values, *args, **kwargs)

        return recorded

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PolicyEngineUKEngine, "materialize", counted)
        for stage, (module, name) in _CORRECTIONS.items():
            patch.setattr(module, name, recording(getattr(module, name), stage))
        return {
            clones: _run(tmp_path_factory.mktemp(f"k{clones}"), clones, pending)
            for clones in (1, _CLONES)
        }


# --- helpers -------------------------------------------------------------------------


def _weights(frame: Frame) -> pd.Series:
    return pd.Series(
        np.asarray(frame.weights_for("household").values, dtype=float),
        index=frame.table("household")["household_id"].to_numpy(),
    )


def _clone_zero(frame: Frame, entity: str) -> pd.DataFrame:
    table = frame.table(entity)
    return table.loc[table[ladder_clone_index_column(entity)].eq(0)]


def _clone_zero_ids(frame: Frame) -> set[int]:
    return set(_clone_zero(frame, "household")["household_id"].tolist())


def _copied(frame: Frame, clone: int) -> set[int]:
    """The households spi_support_channel copied in ``clone``, by the copies'
    source ids."""

    household = frame.table("household")
    rows = household[support_channel_column("household")].eq(
        SPI_SYNTHETIC_SUPPORT_CHANNEL
    ) & household[ladder_clone_index_column("household")].eq(clone)
    return set(household.loc[rows, support_source_id_column("household")].tolist())


def _seating(frame: Frame, clone: int) -> dict[int, float]:
    """The band donors' seats in ``clone``: each donor's source household, and
    its band."""

    household = frame.table("household")
    donors = household.loc[
        household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]
        & household[ladder_clone_index_column("household")].eq(clone)
    ]
    return dict(
        zip(
            donors[support_source_id_column("household")].tolist(),
            donors[SPI_INCOME_BAND_DONOR_LOWER_BOUND_COLUMN].tolist(),
            strict=True,
        )
    )


def _carriers(frame: Frame, clone: int) -> set[int]:
    """The band donors' carriers in ``clone``, by the person each copy was made
    from."""

    person = frame.table("person")
    carriers = person.loc[
        person[PERSON_IS_SPI_INCOME_BAND_CARRIER]
        & person[ladder_clone_index_column("person")].eq(clone)
    ]
    return set(carriers[support_source_id_column("person")].tolist())


def _copies_off_their_parents_clone(frame: Frame) -> int:
    """How many households carry another clone index than the household
    their support lineage names, which spi_support_channel and
    spi_income_band_donors keep beside their copies (a household that is no
    copy names itself)."""

    household = frame.table("household")
    index = ladder_clone_index_column("household")
    by_id = household.set_index("household_id")[index]
    parents = household[support_source_id_column("household")].map(by_id)
    return int((parents.isna() | (parents.to_numpy() != household[index])).sum())


def _moved(run: _Run, stage: str) -> frozenset[int]:
    """The households ``stage``'s exact-total correction moved: where its
    output weights differ from the weights it was handed."""

    (values,) = run.handed[stage]
    weights = _weights(run.frames[stage])
    return frozenset(weights.index[weights.to_numpy() != values].tolist())


def _corrected(one: _Run, many: _Run, stage: str) -> set[int]:
    """The households an exact-total correction moved, by spi_support_channel
    or ``stage``, at either clone count."""

    return {
        household_id
        for run in (one, many)
        for earlier in (SPI, stage)
        if earlier in run.handed
        for household_id in _moved(run, earlier)
    }


def _nesting_failures(one: _Run, many: _Run, stage: str) -> tuple[Counter, Counter]:
    """K x each clone-0 weight against one clone's, at ``stage``'s version, in
    two classes at a power-of-two K: rows an exact-total correction moved
    (``_corrected``) within ``rows x K x 2.2e-16`` relative, every other row
    bitwise. Returns the classes' counts and their failures'."""

    reference = _weights(one.frames[stage])
    clone_zero = _clone_zero_ids(many.frames[stage])
    assert clone_zero == set(reference.index), (
        f"Clone 0 at K={many.clones} after {stage} is not the one-clone "
        "frame's households."
    )
    scaled = _weights(many.frames[stage]).loc[reference.index]
    corrected = _corrected(one, many, stage)
    bound = Fraction(len(reference) * many.clones) * Fraction(_EPSILON)
    classes: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for household_id, weight, expected in zip(
        reference.index.tolist(), scaled.tolist(), reference.tolist(), strict=True
    ):
        difference = abs(Fraction(weight) * many.clones - Fraction(expected))
        if household_id in corrected:
            classes["corrected"] += 1
            if difference > bound * Fraction(expected):
                failures["corrected rows beyond rows x K x 2.2e-16 relative"] += 1
        else:
            classes["bitwise"] += 1
            if difference:
                failures["uncorrected rows not bitwise at a power-of-two K"] += 1
    return classes, failures


# --- every stage, every engine call ---------------------------------------------------


def test_every_stage_runs_and_every_engine_call_after_g1_runs_clone_by_clone(
    runs,
) -> None:
    one, many = runs[1], runs[_CLONES]
    assert one.stage_names == many.stage_names
    assert len(many.stage_names) == _STAGE_COUNT
    assert one.order == many.order
    for run in (one, many):
        assert set(run.hits) == set(run.order) == set(run.calls)
        assert not any(run.hits.values()), f"A node hit its record at K={run.clones}."
    first = many.order.index(G1)
    called = [node for node in many.order if many.calls[node]]
    after = [node for node in called if many.order.index(node) >= first]
    assert {_LAST_DRAW, "private_school_attendance"} <= set(after)
    # The area stages before the draws and private_rent call no engine, and
    # neither does spi_copy_areas on a fixture without a Northern Ireland
    # copy, the one its call reads.
    assert not {*_AREA_STAGES, COPY_AREAS, PRIVATE_RENT_STAGE_NAME} & set(called)
    for node in called:
        # As many calls as at one clone: no stage calls the engine per clone.
        assert len(many.calls[node]) == len(one.calls[node]), node
        assert set(one.calls[node]) == {1}, node
        expected = _CLONES if many.order.index(node) >= first else 1
        assert set(many.calls[node]) == {expected}, (node, expected)
    assert set(called) == {node for node in one.order if one.calls[node]}
    # private_school_attendance restates attendance over the pool when its
    # call runs in batches, and keeps the engine's own at one clone.
    routes = {
        run.clones: run.evidence["private_school_attendance"]["rank_route"]
        for run in (one, many)
    }
    assert routes == {
        1: PRIVATE_SCHOOL_ATTENDANCE_ENGINE_ROUTE,
        _CLONES: PRIVATE_SCHOOL_ATTENDANCE_POOLED_ROUTE,
    }


def test_every_population_keeps_the_clone_index_invariant(runs) -> None:
    for run in runs.values():
        broken = {node: found for node, found in run.problems.items() if found}
        assert not broken, f"At K={run.clones}: {broken}."
        for stage in (SPI, DONORS):
            off = _copies_off_their_parents_clone(run.frames[stage])
            assert off == 0, (
                f"{off} household(s) after {stage} at K={run.clones} carry another "
                "clone index than the household their support lineage names."
            )


def test_every_node_keeps_the_one_clone_household_mass(runs) -> None:
    """The clones split each household's weight, and every later stage
    conserves or reallocates mass as at one clone: each node's household
    mass is K = 1's, summed in another order."""

    one, many = runs[1], runs[_CLONES]
    off = [
        node
        for node in many.order
        if not math.isclose(many.masses[node], one.masses[node], rel_tol=_SUM_RTOL)
    ]
    assert not off, f"{len(off)} node(s) at K={_CLONES} change the household mass."


# --- exact ------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", (G1, G1B))
def test_clone_zero_after_the_clones_is_the_one_clone_frame(runs, stage: str) -> None:
    """Clone-major: clone 0 is the frame's first rows, every column and dtype
    as at one clone (the ids, G1's columns and G1b's key among them), at K
    times its weight bit for bit, since 1/4 splits exactly."""

    one, many = runs[1].frames[stage], runs[_CLONES].frames[stage]
    written = {*UK_GEOGRAPHIC_EXPAND_OUTPUT_COLUMNS, GEOGRAPHY_HOUSEHOLD_KEY_COLUMN}
    for entity in UK_NATIONAL_SCHEMA.entities:
        clone_zero = _clone_zero(many, entity)
        assert clone_zero.index.tolist() == list(range(len(one.table(entity))))
        pd.testing.assert_frame_equal(clone_zero, one.table(entity), check_exact=True)
        written -= set(one.table(entity).columns)
    if stage == G1:
        written.discard(GEOGRAPHY_HOUSEHOLD_KEY_COLUMN)
    assert not written, f"Some of {stage}'s columns were not compared."
    weights = _weights(many).loc[_weights(one).index].to_numpy()
    assert (weights * _CLONES).tobytes() == _weights(one).to_numpy().tobytes()


def test_clone_zero_through_area_brma_is_the_one_clone_frame(runs) -> None:
    """Through area_brma clone 0 is the one-clone frame on every column, every
    area and every draw from atomic_area_assign to area_brma among them:

    - atomic_area_assign keys each row's draw on its geography key, which
      clone 0 shares with one clone, at the fallback level its source's own
      decile and band give it; local_authority and rfdr read the drawn area.
    - frs_take_up's anchored residual and frs_person_draws' apprentice walk
      count within each clone (``_COUNTED_WITHIN_EACH_CLONE``); the walk
      compares prefixes of clone 0's weights, one clone's over 4, which the
      split scales exactly.
    - Every other draw is keyed on the row's own id, which clone 0 shares
      with one clone.
    - area_brma reads each Great Britain household's BRMA off its drawn area;
      a Northern Ireland household's would take its count-table cell from the
      engine's LHA_category, which the batched engine computes for clone 0
      from that clone's rows alone, which are one clone's.
    """

    one, many = runs[1].frames[_LAST_DRAW], runs[_CLONES].frames[_LAST_DRAW]
    compared: set[tuple[str, str]] = set()
    differing: list[str] = []
    for entity in UK_NATIONAL_SCHEMA.entities:
        reference = one.table(entity).reset_index(drop=True)
        clone_zero = _clone_zero(many, entity).reset_index(drop=True)
        assert list(clone_zero.columns) == list(reference.columns), entity
        for column in reference.columns:
            compared.add((entity, column))
            if not clone_zero[column].equals(reference[column]):
                differing.append(f"{entity}.{column}")
    assert not differing, f"Clone 0 after {_LAST_DRAW} differs on {differing}."
    stages = runs[1].stage_names
    drawn = {
        (entity, column)
        for stage in (
            *_AREA_STAGES,
            _FIRST_DRAW,
            "frs_person_draws",
            "frs_household_draws",
            _LAST_DRAW,
        )
        for column in _stage_outputs(stages, stage)
        for entity in UK_NATIONAL_SCHEMA.entities
        if column in one.table(entity).columns
    }
    areas = {
        ("household", column)
        for stage in _AREA_STAGES
        for column in _stage_outputs(stages, stage)
    }
    assert areas
    assert _COUNTED_WITHIN_EACH_CLONE | {_BRMA} | areas <= drawn <= compared


def test_clone_zero_s_copies_take_the_one_clone_copies_keys_and_areas(runs) -> None:
    """spi_copy_areas keys each SPI copy over its own path, which clone 0's
    copies share with one clone's, and draws its area on that key: clone 0's
    households carry K = 1's key, areas, local authority, BRMA and
    fuel-relief flag on every row, the copies' rewrites among them, and some
    other clone's copies take other keys."""

    one, many = runs[1].frames[COPY_AREAS], runs[_CLONES].frames[COPY_AREAS]
    reference = one.table("household").set_index("household_id")
    clone_zero = _clone_zero(many, "household").set_index("household_id")
    assert set(clone_zero.index) == set(reference.index)
    copies = reference.index[
        reference[support_channel_column("household")].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)
    ]
    assert len(copies)
    differing = [
        column
        for column in UK_SPI_COPY_AREAS_REWRITE_COLUMNS
        if not clone_zero.loc[reference.index, column].equals(reference[column])
    ]
    assert not differing, f"Clone 0 after {COPY_AREAS} differs on {differing}."
    household = many.table("household")
    keys = household.set_index("household_id")["geography_household_key"]
    assert keys.is_unique
    later = household.loc[
        household[support_channel_column("household")].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)
        & household[ladder_clone_index_column("household")].gt(0),
        "geography_household_key",
    ]
    assert not set(later) & set(reference.loc[copies, "geography_household_key"])


def _stage_outputs(stage_names: tuple[str, ...], stage: str) -> tuple[str, ...]:
    from microcosm.build.country_spec import load_country_spec

    assert stage in stage_names
    declared = load_country_spec("uk").sources.stage_map()[stage]
    return (*declared.outputs, *declared.rewrites)


def test_the_support_channel_copies_the_one_clone_households_in_clone_zero(
    runs,
) -> None:
    one, many = runs[1], runs[_CLONES]
    reference = _copied(one.frames[SPI], 0)
    assert reference
    assert _copied(many.frames[SPI], 0) == reference
    assert any(
        _copied(many.frames[SPI], clone) != reference for clone in range(1, _CLONES)
    )
    classes, failures = _nesting_failures(one, many, SPI)
    assert not failures, f"After {SPI} at K={_CLONES} ({dict(classes)}): {failures}."


def test_the_band_donors_seat_the_one_clone_households_in_clone_zero(runs) -> None:
    one, many = runs[1].frames[DONORS], runs[_CLONES].frames[DONORS]
    # Clone 0 holds exactly the one-clone frame's households, the support
    # channel's copies and the donors among them.
    assert _clone_zero_ids(many) == set(_weights(one).index)
    seating = _seating(one, 0)
    assert seating
    assert _seating(many, 0) == seating
    assert _carriers(many, 0) == _carriers(one, 0)
    seats = Counter(seating.values())
    for clone in range(_CLONES):
        assert Counter(_seating(many, clone).values()) == seats, clone


# --- the band donors' weights ---------------------------------------------------------


def _mass_scales(runs) -> tuple[float, float, str]:
    one = float(runs[1].evidence[DONORS]["mass_scale"])
    many = float(runs[_CLONES].evidence[DONORS]["mass_scale"])
    report = (
        f"{DONORS}'s mass scale at K={_CLONES}: equal to K=1's {many == one}, "
        f"below it {many < one}"
    )
    return one, many, report


def test_the_band_donors_report_their_mass_scale_and_scale_clone_zero_donors_by_it(
    runs,
) -> None:
    """Reported first: spi_income_band_donors' mass scale at one clone and at
    four. Every clone's donors take one scale, the smallest any clone's region
    allows, so each clone-0 donor weighs one clone's donor weight times the
    ratio of the two scales, to the rounding of the two products (a few
    ulps)."""

    scale_one, scale_many, report = _mass_scales(runs)
    one, many = runs[1].frames[DONORS], runs[_CLONES].frames[DONORS]
    household = one.table("household").set_index("household_id")
    donors = household.index[household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR]]
    assert len(donors), report
    left = _CLONES * _weights(many).loc[donors].to_numpy() * scale_one
    right = _weights(one).loc[donors].to_numpy() * scale_many
    off = int((np.abs(left - right) > 4 * np.spacing(right)).sum())
    assert off == 0, f"{off} clone-0 donor(s) off the scales' ratio; {report}."


def test_the_band_donors_fund_each_clone_zero_stratum_at_one_ratio(runs) -> None:
    """K x clone-0 weights after spi_income_band_donors, over one clone's.
    The stage scales every incumbent of a funding stratum by the stratum's
    one factor, so in each stratum of clone 0 every incumbent takes one
    ratio, the stratum's factor at K = 4 over its factor at one clone, to the
    rounding of the two products (a few ulps), apart from the rows an
    exact-total correction moved; a stratum without donors takes exactly
    one, both its factors being one; and where the two mass scales are
    equal, every ratio is one, so K x clone-0 weights nest. On the H2
    fixture's capped rung the scale at K = 4 is below one clone's, so the
    strata with donors take ratios other than one. Whether the stage takes a
    mass scale per clone is open (microcosm#931)."""

    one, many = runs[1], runs[_CLONES]
    scale_one, scale_many, report = _mass_scales(runs)
    reference = _weights(one.frames[DONORS])
    ratio = (
        _CLONES * _weights(many.frames[DONORS]).loc[reference.index] / reference
    ).to_numpy()
    household = many.frames[DONORS].table("household")
    by_id = household["household_id"].to_numpy()
    strata = (
        pd.Series(spi_band_donors._funding_strata(household).to_numpy(), index=by_id)
        .loc[reference.index]
        .to_numpy()
    )
    donor = (
        pd.Series(
            household[HOUSEHOLD_IS_SPI_INCOME_BAND_DONOR].to_numpy(dtype=bool),
            index=by_id,
        )
        .loc[reference.index]
        .to_numpy()
    )
    corrected = reference.index.isin(list(_corrected(one, many, DONORS)))
    funded = set(strata[donor])
    ratios: dict[str, float] = {}
    off: Counter[str] = Counter()
    for stratum in sorted(set(strata[~donor])):
        values = ratio[(strata == stratum) & ~donor & ~corrected]
        if not values.size:
            continue
        ratios[stratum] = float(values[0])
        if (np.abs(values - values[0]) > 8 * np.spacing(values[0])).any():
            off["strata whose incumbents take more than one ratio"] += 1
        if stratum not in funded and not (values == 1.0).all():
            off["strata without donors off ratio one"] += 1
    assert funded and funded <= set(ratios), report
    if scale_many == scale_one and any(
        abs(value - 1.0) > 8 * np.spacing(1.0) for value in ratios.values()
    ):
        off["strata off ratio one at equal mass scales"] += 1
    assert not off, f"After {DONORS} at K={_CLONES}: {dict(off)}; {report}."


# --- tolerance classes ---------------------------------------------------------------


def test_the_draws_keep_their_k_invariant_receipt_fields(runs) -> None:
    """frs_take_up, frs_person_draws and frs_household_draws count each
    flag's population in true sources, and frs_person_draws' apprentice walk
    its target, masses and supports, so these are one clone's at K = 4; the
    flagged support too, counted in clone 0, whose walk is one clone's. Each
    target and mass, summed over the clones' rows, is one clone's to its
    summation order. The realized shares and mass are left to the share
    class: the other clones flag their own employees."""

    one, many = runs[1].evidence, runs[_CLONES].evidence
    for stage in (_FIRST_DRAW, "frs_person_draws", "frs_household_draws"):
        flags = {
            run: {
                row["output"]: row for row in evidence[stage]["take_up_signal"]["flags"]
            }
            for run, evidence in (("one", one), ("many", many))
        }
        assert set(flags["one"]) == set(flags["many"]), stage
        for output, row in flags["one"].items():
            other = flags["many"][output]
            for field in ("population_support", "distinct_values", "rate_key"):
                assert other.get(field) == row.get(field), (stage, output, field)
            if "target" in row:
                assert math.isclose(other["target"], row["target"], rel_tol=_SUM_RTOL)
    apprentices = {
        run: evidence["frs_person_draws"]["apprentice_receipt"]
        for run, evidence in (("one", one), ("many", many))
    }
    for field in (
        "target",
        "rate",
        "population_mass",
        "eligible_mass",
        "population_support",
        "eligible_support",
        "flagged_support",
        "cap_bound",
    ):
        expected, found = apprentices["one"][field], apprentices["many"][field]
        if isinstance(expected, float) and isinstance(found, float):
            assert math.isclose(found, expected, rel_tol=_SUM_RTOL), field
        else:
            assert found == expected, field


def test_the_area_stages_receipts_are_one_clone_s_but_for_k(runs) -> None:
    """The area stages, the SPI copies' areas and the private rents count
    distinct true sources, over every clone where each count is the same at
    every K and among clone-0 rows where a union could grow with K
    (microcosm#931), so every field of their receipts but the clone count is
    one clone's at K = 4: the counts, the levels and fallback reasons, the
    booleans, the digests, the seed and the stream."""

    one, many = runs[1].evidence, runs[_CLONES].evidence
    for stage in _K_INVARIANT_RECEIPTS:
        assert one[stage]["clone_count"] == 1, stage
        assert many[stage]["clone_count"] == _CLONES, stage
        same = {
            name: value for name, value in many[stage].items() if name != "clone_count"
        }
        reference = {
            name: value for name, value in one[stage].items() if name != "clone_count"
        }
        differing = sorted(
            name
            for name in set(same) | set(reference)
            if same.get(name) != reference.get(name)
        )
        assert not differing, (
            f"{stage}'s receipt at K={_CLONES} differs on {differing}."
        )


def test_every_later_output_is_carried_by_every_clone_at_a_nearby_share(
    runs,
) -> None:
    """From frs_take_up on, every stage's declared outputs (frs_take_up's
    residual, frs_person_draws' walk, cgt_support_split and everything
    downstream of an engine call among them) at K = 4: a weighted share over
    the clones within
    ``_SHARE_TOLERANCE`` of one clone's, and on every clone wherever one
    clone gives it ``_PRESENCE_ROWS`` rows or more."""

    one, many = runs[1].shares, runs[_CLONES].shares
    assert set(one) == set(many)
    assert {stage for stage, _ in one} >= {
        _FIRST_DRAW,
        "frs_person_draws",
        "cgt_support_split",
        "private_school_attendance",
    }
    apart = sorted(
        f"{stage}.{column}"
        for (stage, column), (share, _) in one.items()
        if abs(many[(stage, column)][0] - share) > _SHARE_TOLERANCE
    )
    assert not apart, f"Outputs whose share at K={_CLONES} is apart: {apart}."
    absent = sorted(
        f"{stage}.{column}"
        for (stage, column), (_, (rows,)) in one.items()
        if rows >= _PRESENCE_ROWS and min(many[(stage, column)][1]) == 0
    )
    assert not absent, f"Outputs a clone at K={_CLONES} does not carry: {absent}."


# --- areas and keys ---------------------------------------------------------------------


def _area_columns(stage_names: tuple[str, ...]) -> tuple[str, ...]:
    """The household columns the area stages and area_brma write."""

    return tuple(
        column
        for stage in (*_AREA_STAGES, _LAST_DRAW)
        for column in _stage_outputs(stage_names, stage)
    )


def _keys_by_id(frame: Frame) -> pd.Series:
    household = frame.table("household")
    return pd.Series(
        household[GEOGRAPHY_HOUSEHOLD_KEY_COLUMN].to_numpy(dtype=object),
        index=household["household_id"].to_numpy(),
    )


@pytest.mark.parametrize("clones", (1, _CLONES))
def test_on_h2_every_household_has_an_area_from_area_brma_on(runs, clones) -> None:
    """After area_brma, after spi_copy_areas and in the spine's last
    population, every household row, the SPI copies, the band donors and
    every later stage's copies among them, carries a drawn area and its
    ladder, a local authority, a fuel-relief flag and a BRMA (microcosm#931)."""

    run = runs[clones]
    columns = _area_columns(run.stage_names)
    assert {
        "atomic_area_code",
        "local_authority",
        "in_rural_fuel_duty_relief_area",
        "brma",
    } <= set(columns)
    for stage in (_LAST_DRAW, COPY_AREAS, _FINAL):
        household = run.frames[stage].table("household")
        lacking = sorted(
            column
            for column in columns
            if column not in household.columns
            or household[column].isna().any()
            or (
                not pd.api.types.is_bool_dtype(household[column].dtype)
                and household[column].astype(str).eq("").any()
            )
        )
        assert not lacking, f"After {stage} at K={clones} a household lacks {lacking}."


@pytest.mark.parametrize("clones", (1, _CLONES))
def test_on_h2_every_spi_channel_row_takes_a_key_of_its_own(runs, clones) -> None:
    """The geography keys are unique from geographic_identity through
    area_brma. spi_support_channel's copies and the band donors carry the key
    of the household they copy until spi_copy_areas gives every SPI-channel
    row a key of its own and leaves every other row's, so the keys are unique
    again after it. Later stages keep each row's key and give their copies
    keys spi_copy_areas gave, their parents' (microcosm#931)."""

    run = runs[clones]
    for stage in (G1B, _LAST_DRAW):
        assert _keys_by_id(run.frames[stage]).is_unique, f"{stage} at K={clones}"
    base = set(_keys_by_id(run.frames[_LAST_DRAW]))
    household = run.frames[DONORS].table("household")
    channel = (
        household[support_channel_column("household")]
        .eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)
        .to_numpy()
    )
    assert channel.any() and not channel.all()
    ids = household["household_id"].to_numpy()
    before = _keys_by_id(run.frames[DONORS])
    after = _keys_by_id(run.frames[COPY_AREAS])
    assert set(after.index) == set(ids)
    # Before the copy stage each SPI-channel row repeats a key the geography
    # identity gave a household it copies.
    assert set(before.loc[ids[channel]]) <= base
    # After it every SPI-channel row has a key of its own, every other row
    # keeps its key, and the keys are unique.
    assert after.is_unique, f"K={clones}"
    assert not set(after.loc[ids[channel]]) & base, f"K={clones}"
    assert after.loc[ids[~channel]].equals(before.loc[ids[~channel]]), f"K={clones}"
    # Later stages keep each row's key; their copies carry keys the copy
    # stage gave.
    final = _keys_by_id(run.frames[_FINAL])
    earlier = final.index.isin(after.index)
    assert final[earlier].equals(after.loc[final.index[earlier]]), f"K={clones}"
    assert set(final[~earlier]) <= set(after), f"K={clones}"


@pytest.mark.parametrize("clones", (1, _CLONES))
def test_on_h2_private_rent_rewrites_only_great_britain_private_renters(
    runs, clones
) -> None:
    """private_rent re-draws the rent of every row it reads as a Great
    Britain private renter with a rent, and leaves every other row's bit for
    bit; its receipt counts the re-drawn rows' clone-0 sources
    (microcosm#931). H2's root households all own their homes, so the stage
    re-draws no row here: test_uk_spine_three_systems.py re-draws renters on
    the BRMAs the spine's own stages give them."""

    run = runs[clones]
    before = run.frames[_HOUSING].table("household")
    after = run.frames[PRIVATE_RENT_STAGE_NAME].table("household")
    assert after["household_id"].tolist() == before["household_id"].tolist()
    redrawn = prl.redraw_mask(
        before["region"].to_numpy(dtype=object),
        before["tenure_type"].to_numpy(dtype=object),
        before["rent"].to_numpy(dtype=float),
    )
    old = before["rent"].to_numpy(dtype=np.float64).view(np.int64)
    same = old == after["rent"].to_numpy(dtype=np.float64).view(np.int64)
    assert same[~redrawn].all(), f"K={clones}"
    assert not same[redrawn].any(), f"K={clones}"
    # The housing shell draws the SPI copies' tenures from a classifier it
    # trains on the FRS households, all owners on H2, so H2 holds no row to
    # re-draw; a fixture that gains one gets its re-draw checked above.
    assert not redrawn.any(), f"H2 now holds a private renter to re-draw (K={clones})."
    clone_zero = before[ladder_clone_index_column("household")].eq(0).to_numpy()
    sources = source_household_ids(before, stage=PRIVATE_RENT_STAGE_NAME)
    assert run.evidence[PRIVATE_RENT_STAGE_NAME]["source_households_redrawn"] == (
        rounded_count(distinct_carriers(sources[redrawn & clone_zero].tolist()))
    )


@pytest.mark.parametrize("clones", (1, _CLONES))
def test_on_h2_the_wealth_chain_localises_every_england_and_wales_owner(
    runs, clones
) -> None:
    """After was_wealth every England and Wales owner with a positive draw
    holds the value the area home values read off its constituency's knots,
    with the draw kept beside it, and every other row keeps the draw bit for
    bit (G8, form F); the stage's block counts the rewritten sources of each
    knots region H2 holds, at its clone count. The draw reaches the spine's
    last population and the release boundary drops it (microcosm#931). H2's
    root holds Scottish owners, whose draws stay. At K = 4 the rank pools over
    the clones, so the values nest on one clone's only to the share class."""

    run = runs[clones]
    household = run.frames[_WAS].table("household")
    basis = household[RANK_BASIS_COLUMN].to_numpy(dtype=np.float64)
    value = household["main_residence_value"].to_numpy(dtype=np.float64)
    region = household["region"].astype(str).to_numpy()
    england_wales = np.isin(region, list(FRS_REGION_TO_KNOTS_REGION))
    owners = owner_mask(household["tenure_type"].to_numpy()) & (basis > 0)
    localised = owners & england_wales
    assert localised.any() and (owners & ~england_wales).any(), f"K={clones}"
    moved = value.view(np.int64) != basis.view(np.int64)
    assert np.array_equal(moved, localised), (
        f"After {_WAS} at K={clones} the rows whose main residence moved off "
        "the draw are not England and Wales's owners."
    )
    block = run.evidence[_WAS][AREA_HOME_VALUES_EVIDENCE]
    assert block["clone_count"] == clones
    assert block["knots_level"] == "constituency"
    assert set(block["sources_rewritten_by_region"]) == {
        FRS_REGION_TO_KNOTS_REGION[name] for name in set(region[localised])
    }
    final = run.frames[_FINAL]
    assert RANK_BASIS_COLUMN in final.table("household").columns
    released = uk_release_export_frame(final).table("household")
    assert RANK_BASIS_COLUMN not in released.columns


def _same_column(left: pd.Series, right: pd.Series) -> bool:
    """Whether two columns hold the same values row for row, by position:
    floats bit for bit (or equal, missing values alike), anything else as
    objects with a missing value as None."""

    if pd.api.types.is_float_dtype(left.dtype) and pd.api.types.is_float_dtype(
        right.dtype
    ):
        a = left.to_numpy(dtype=np.float64)
        b = right.to_numpy(dtype=np.float64)
        return bool(
            np.array_equal(a.view(np.int64), b.view(np.int64))
            or np.array_equal(a, b, equal_nan=True)
        )
    a = left.astype(object).where(left.notna(), None).to_numpy()
    b = right.astype(object).where(right.notna(), None).to_numpy()
    return bool((a == b).all())


@pytest.mark.parametrize("clones", (1, _CLONES))
def test_on_h2_the_uprating_lands_localised_homes_and_copies_keep_them(
    runs, clones
) -> None:
    """At one clone and at four (microcosm#931): was_wealth's main residences
    are the area home values on the frame the stage ran on, at its household
    weights and the draw it kept, bit for bit; regional_property_uprating,
    dividing by that draw, returns each localised home to the knot its
    owner's rank reads, within one rounding, and a home at the knots' cap to
    the cap or a few roundings below, never above; and every high-value
    support copy equals the row the copy-id table names its parent on every
    column but its ids and the stage's own three, its value, kept draw, area
    and geography key among them (C-19). H2 parity cannot see these laws: its
    oracle and the graph run the same code."""

    run = runs[clones]
    resources = load_committed_area_home_values_resources()
    before = run.frames[PRIVATE_RENT_STAGE_NAME]
    household = before.table("household").reset_index(drop=True)
    order = household["household_id"].to_numpy()
    weights = np.asarray(before.weights_for("household").values, dtype=np.float64)
    was = run.frames[_WAS].table("household").set_index("household_id")
    drawn = was.loc[order, RANK_BASIS_COLUMN].to_numpy(dtype=np.float64)
    stored = was.loc[order, "main_residence_value"].to_numpy(dtype=np.float64)
    direct = localise_area_home_values(household, weights, drawn, resources=resources)
    assert np.array_equal(stored.view(np.int64), direct.value.view(np.int64)), (
        f"{_WAS} at K={clones} does not hold the area home values on its input."
    )
    tenure = household["tenure_type"].to_numpy()
    rank, _ = owner_uniforms(
        spine_key=household["household_id"].astype(str).to_numpy(dtype=object),
        region=household["region"].astype(str).to_numpy(),
        tenure_type=tenure,
        value=drawn,
        weight=weights,
        stream=AREA_HOME_VALUES_STREAM,
    )
    knot, rewritten = localise_main_residence_values(
        u=rank,
        constituency_code=household["constituency_code"]
        .to_numpy(dtype=object)
        .astype(str),
        tenure_type=tenure,
        value=drawn,
        knots=resources.knots,
    )
    assert rewritten.any() and np.array_equal(rewritten, direct.rewritten)
    uprated = (
        run.frames[_UPRATING]
        .table("household")
        .set_index("household_id")
        .loc[order, "main_residence_value"]
        .to_numpy(dtype=np.float64)
    )
    capped = rewritten & (knot >= resources.knots.cap)
    plain = rewritten & ~capped
    assert np.all(np.abs(uprated[plain] - knot[plain]) <= np.spacing(knot[plain])), (
        f"{_UPRATING} at K={clones} leaves a localised home off its knot."
    )
    assert np.all(uprated[capped] <= knot[capped]) and np.all(
        uprated[capped] >= knot[capped] - 4 * np.spacing(knot[capped])
    ), f"{_UPRATING} at K={clones} returns a home at the cap past it."
    split = (
        run.frames[_HIGH_VALUE].table("household").set_index("household_id", drop=False)
    )
    copies = split[split[HIGH_VALUE_SUPPORT_COPY_COLUMN].astype(bool)]
    assert len(copies), f"H2 holds no high-value support copy at K={clones}."
    parents = copy_id_table.parent(
        copies["household_id"].to_numpy(np.int64), HIGH_VALUE_SUPPORT_COPY_ID_FIELD
    )
    owned = {
        "household_id",
        HIGH_VALUE_SUPPORT_COPY_COLUMN,
        HIGH_VALUE_SUPPORT_INDEX_COLUMN,
        HIGH_VALUE_SUPPORT_COPIES_COLUMN,
    }
    off = [
        column
        for column in split.columns
        if column not in owned
        and not _same_column(copies[column], split.loc[parents, column])
    ]
    assert not off, (
        f"High-value support copies at K={clones} differ from their parents on {off}."
    )


# --- receipts ---------------------------------------------------------------------------


def _found(value: object, key: str) -> list[object]:
    """Every value stored under ``key`` anywhere in ``value``."""

    found: list[object] = []
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name == key:
                found.append(item)
            found.extend(_found(item, key))
    elif isinstance(value, list | tuple):
        for item in value:
            found.extend(_found(item, key))
    return found


def test_the_receipts_carry_the_clone_count_and_the_fold_receipts(runs) -> None:
    one, many = runs[1].evidence, runs[_CLONES].evidence
    for stage in (G1, SPI, DONORS):
        assert one[stage]["clone_count"] == 1, stage
        assert many[stage]["clone_count"] == _CLONES, stage
    # The counts in true sources are one clone's.
    assert many[G1]["source_households"] == one[G1]["source_households"]
    for field in ("expected_spi_households", "source_households"):
        assert many[SPI][field] == one[SPI][field], field
    for field in ("eligible_households", "seating_scale"):
        assert many[DONORS][field] == one[DONORS][field], field
    for evidence in (one, many):
        assert evidence[DONORS]["funding_within_share"] is True
        assert evidence[DONORS]["funding_matches_bands"] is True
    # The area home values' block in was_wealth's receipt carries its clone
    # count at every K, as the area stages' receipts do (microcosm#931), so the
    # support clip's count is read beside it.
    counted = {
        clones: {
            stage
            for stage, document in evidence.items()
            if document is not None
            and _found(
                {
                    name: item
                    for name, item in document.items()
                    if name != AREA_HOME_VALUES_EVIDENCE
                },
                "clone_count",
            )
        }
        for clones, evidence in ((1, one), (_CLONES, many))
    }
    for clones, evidence in ((1, one), (_CLONES, many)):
        assert evidence[_WAS][AREA_HOME_VALUES_EVIDENCE]["clone_count"] == clones
    assert counted[_CLONES] - counted[1] >= _CLONE_SCALED_RECEIPTS | _FOLDING_STAGES
    assert not counted[1] & (_CLONE_SCALED_RECEIPTS | _FOLDING_STAGES)
    for stage in counted[_CLONES]:
        assert set(_found(many[stage], "clone_count")) == {_CLONES}, stage
    folds = {
        clones: {
            stage: _found(document, GEOGRAPHIC_FOLD_EVIDENCE)
            for stage, document in evidence.items()
            if document is not None and _found(document, GEOGRAPHIC_FOLD_EVIDENCE)
        }
        for clones, evidence in ((1, one), (_CLONES, many))
    }
    assert folds[1] == {}
    assert set(folds[_CLONES]) == _FOLDING_STAGES
    for stage, receipts in folds[_CLONES].items():
        for receipt in receipts:
            assert receipt["clone_count"] == _CLONES, stage
            assert {"records", "varied_records", "varied_share"} <= set(receipt), stage
