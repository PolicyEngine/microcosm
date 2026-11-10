"""Each UK gate battery phase's own code, and the check that refuses a stale one.

A gate battery's phases run one kernel's code, and that kernel's hash is the
code they share. What only one phase's gates compute through is that phase's
own code, a source hash its node carries as its ``<phase>_code`` parameter,
so no other phase's own key carries it (microcosm#931 ruling 6): the spine
gates' assembled and transferred phases (:data:`UK_SPINE_GATE_PHASE_CODE`)
and the full build's preflight and calibrated gates
(:data:`UK_FULL_GATE_PHASE_CODE`).

The four functions and two tables lived in ``graph_evidence`` and
``graph_terminal``, which define the gate kernels and which every spine
stage kernel, the dense population kernels and the export kernels hash too,
so an edit to a phase's list of modules re-keyed every node whose kernel
hashes either file. They moved here unchanged in what they hash, with the
one check both batteries call (microcosm#931 ruling 3a). Only the graph
construction, which sets each node's parameter, and that check read this
module. The check is listed refusal-only in test_uk_graph, and no kernel
hashes this module: a phase's own code reaches its key through the
parameter, which node_key hashes, and both gate kernels' runs call
:func:`require_current_phase_code` before they evaluate anything, so a node
whose parameter differs from the code this process computes is refused
(microcosm#931 ruling 3b, for the spine gates).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from types import MappingProxyType

from microcosm.graph import source_hash


@cache
def uk_spine_assembled_only_code_hash() -> str:
    """The source hash of the code only the assembled spine phase computes
    through: the ONS household type domain its enum gate checks
    (``frs_relationships``), which no transferred-phase gate reads.

    So the transferred phase's own key leaves it out, though an edit to it
    still re-keys the transferred gate, through the ``frs_relationships``
    stage, whose module it is, and every stage after it. Hashed once per
    process, as for the calibrated gates' own code
    (:func:`uk_terminal_only_code_hash`).
    """
    from . import frs_relationships

    return source_hash(frs_relationships)


@cache
def uk_spine_transferred_only_code_hash() -> str:
    """The source hash of the code only the transferred spine phase computes
    through, which the assembled gate's key leaves out.

    That is what the stage-health checks only the transferred phase declares
    compute through: the energy rake's published level and NEED margins
    (``energy_pricing``); the bus travel, bus pricing, bus support pricing,
    consumption basis and gambling checks' published facts and pricing
    (``nts_bus_travel``, ``bus_use_incidence``, ``bus_fare_pricing``,
    ``bus_support_pricing``, ``etb_services``, ``lcfs_consumption``,
    ``donor_uprating``, ``gambling_participation``, the shared ledger's
    calendar-year weights, and the vendored-fact reader with the chronicle
    feed and its schema epochs); the annual income spells check
    (``frs_annual_income``); the apprentice draw's take-up contract
    (``take_up_contract``); and the latent-attribute checks' binomial band
    (``stochastic_assignment``). Its
    enum gate also checks the CGT asset type domain (``cgt_asset_type``).

    The assembled gate's own key, on which every later spine stage's chains,
    therefore carries none of them (the re-review of microcosm#1420, E;
    microcosm#931 ruling 6). An edit to one still re-keys the assembled gate
    wherever a stage before it hashes the module: stages at or before
    ``frs_brma``, the assembled phase's last, hash ``stochastic_assignment``
    and ``take_up_contract``, so an edit to either re-keys every spine node
    after them, the assembled gate included. No stage before that gate hashes
    the other fifteen, so an edit to one of those re-keys the stages that
    hash it, those after them and the transferred gate, never the assembled
    gate. Hashed once per process, as the assembled phase's own code is.
    """
    from .. import chronicle_epoch, stochastic_assignment
    from .. import ledger_targets as shared_ledger_targets
    from . import (
        bus_fare_pricing,
        bus_support_pricing,
        bus_use_incidence,
        cgt_asset_type,
        chronicle_feed,
        donor_uprating,
        energy_pricing,
        etb_services,
        frs_annual_income,
        gambling_participation,
        lcfs_consumption,
        ledger_fact_vendoring,
        nts_bus_travel,
        take_up_contract,
    )

    return source_hash(
        energy_pricing,
        lcfs_consumption,
        bus_fare_pricing,
        bus_support_pricing,
        bus_use_incidence,
        nts_bus_travel,
        etb_services,
        gambling_participation,
        donor_uprating,
        frs_annual_income,
        take_up_contract,
        cgt_asset_type,
        stochastic_assignment,
        chronicle_feed,
        chronicle_epoch,
        ledger_fact_vendoring,
        shared_ledger_targets,
    )


#: Each spine gate phase's own code: the hash of what only that phase's gates
#: compute through, which its node carries as its ``<phase>_code`` parameter
#: and the spine gate kernel checks before it evaluates anything
#: (:func:`require_current_phase_code`), so no other phase's own key carries
#: it. An edit there still re-keys the other phase wherever a node upstream of
#: it hashes the module, as each function's docstring says. The code both
#: phases compute through is the gate kernel's own hash (microcosm#931 ruling
#: 6); test_uk_graph derives each phase's share from the gates it declares.
UK_SPINE_GATE_PHASE_CODE: Mapping[str, Callable[[], str]] = MappingProxyType(
    {
        "assembled": uk_spine_assembled_only_code_hash,
        "transferred": uk_spine_transferred_only_code_hash,
    }
)


@cache
def uk_terminal_only_code_hash() -> str:
    """The source hash of the code only the calibrated gates compute through.

    The source preflight runs the gate kernel's code and the dense solve keys
    on the preflight, so the code only the terminal phase computes through
    keys the calibrated node through its ``terminal_code`` parameter, not the
    kernel's hash (microcosm#1306 review; microcosm#931 ruling 6), and the
    preflight's own key, and with it the solve's, carries none of it. An edit
    to one of these modules re-keys the calibrated gates and what follows
    them, and re-solves too wherever a node upstream of the solve hashes the
    module, as a spine stage, a target or problem kernel, a population kernel
    or the dense solve itself does for most of them. Eight never re-solve, on
    any line: ``high_value_home_gates``, ``microcosm.diagnostics``' catalogs
    and schema, ``calibrate.hierarchy_labels``, the target loss attribution,
    ``input_mass``, ``cgt_projection`` and ``source_runtime``.
    ``uk_runtime.diagnostics`` re-solves only the national line, whose problem
    kernel hashes it. ``calibration_run`` and ``release_certification``
    re-solve only on a bound spine, whose kernel hashes both, and seven that
    upstream of the solve only the spine's stage and gate nodes hash re-solve
    only in the in-graph build (``donor_uprating``, ``etb_services``,
    ``etb_vat``, ``frs_take_up``, ``hmrc_capital_gains``,
    ``take_up_contract``, ``uc_capital_coherence``). That is the terminal
    branch of the full gate kernel's run, and what the terminal gates compute
    through beyond the preflight's:

    - the high-value homes evidence, on the calibrated pool alone
      (``high_value_home_gates``, with the disclosure floor it reads from
      ``receipt_counts`` and the tenure categories behind its owner mask,
      ``tenure_constants``). ``high_value_homes`` itself is in the kernel's
      hash, since the preflight's target-surface evidence reads its metric
      names, and the target kernels hash it too, so an edit to it re-solves:
      a disclosure fix between two builds that must reuse the solve belongs
      in ``high_value_home_gates`` alone. ``calibrate.geography_constants``,
      which the preflight's ledger parity reads, is in the kernel's hash too;
    - the calibration diagnostics the calibrated node publishes as schema 9:
      ``uk_runtime.diagnostics``, with its consumption drift
      (``domestic_energy_targets``, ``energy_pricing``, ``etb_vat``,
      ``lcfs_consumption``) and label choices (``local_authority_input``,
      ``calibrate.hierarchy_labels``); ``microcosm.calibrate``'s
      diagnostics, target loss attribution and solve; and
      ``microcosm.diagnostics``' catalogs and schema;
    - the solved problem and result it decodes (``microcosm.calibrate``'s
      artifacts, matrix and monetary binding), the frame it rebuilds
      (``graph_population``), the area-support summary (``local_rowwise``),
      the support floors, admin anchors and CGT projection it builds the
      gate context with (``support_floor``, ``etb_services``,
      ``cgt_projection``, ``donor_uprating``, and the runner's admin totals
      and projection artifact, ``calibration_run``), the input-mass reference
      (``weighted_integrity``) and the fit-weight records it rehydrates
      (``release_certification``);
    - the terminal gates' evaluators: the input-mass parity
      (``input_mass``), the CGT projection fence's conditioning facts
      (``hmrc_capital_gains``, read through the vendored-fact reader, the
      chronicle feed and its schema epochs), the take-up signal
      (``frs_take_up``, ``take_up_contract``, ``dfe_spring_term``), the
      column implication's property shares (``uc_capital_coherence``), the
      nonnegative columns (``source_runtime``) and the geography ladder
      (``rowwise_geography``).

    Hashed once per process, as the code was imported: an edit to the files
    while a build runs cannot fail the calibrated gates after the solve.
    """
    import microcosm.calibrate._target_loss_attribution as target_loss_attribution
    import microcosm.calibrate.artifacts as calibrate_artifacts
    import microcosm.calibrate.diagnostics as calibrate_diagnostics
    import microcosm.calibrate.hierarchy_labels as hierarchy_labels
    import microcosm.calibrate.matrix as calibrate_matrix
    import microcosm.calibrate.monetary_binding as monetary_binding
    import microcosm.calibrate.solve as calibrate_solve
    import microcosm.diagnostics.catalogs as diagnostics_catalogs
    import microcosm.diagnostics.schema as diagnostics_schema

    from .. import chronicle_epoch, input_mass
    from . import (
        calibration_run,
        cgt_projection,
        chronicle_feed,
        dfe_spring_term,
        diagnostics,
        domestic_energy_targets,
        donor_uprating,
        energy_pricing,
        etb_services,
        etb_vat,
        frs_take_up,
        graph_population,
        high_value_home_gates,
        hmrc_capital_gains,
        lcfs_consumption,
        ledger_fact_vendoring,
        local_authority_input,
        local_rowwise,
        receipt_counts,
        release_certification,
        rowwise_geography,
        source_runtime,
        support_floor,
        take_up_contract,
        tenure_constants,
        uc_capital_coherence,
        weighted_integrity,
    )

    return source_hash(
        high_value_home_gates,
        diagnostics_catalogs,
        diagnostics_schema,
        hierarchy_labels,
        diagnostics,
        calibrate_diagnostics,
        target_loss_attribution,
        calibrate_artifacts,
        calibrate_matrix,
        monetary_binding,
        calibrate_solve,
        graph_population,
        local_rowwise,
        support_floor,
        rowwise_geography,
        release_certification,
        weighted_integrity,
        input_mass,
        calibration_run,
        etb_services,
        etb_vat,
        domestic_energy_targets,
        energy_pricing,
        lcfs_consumption,
        cgt_projection,
        donor_uprating,
        hmrc_capital_gains,
        frs_take_up,
        take_up_contract,
        dfe_spring_term,
        receipt_counts,
        uc_capital_coherence,
        tenure_constants,
        source_runtime,
        local_authority_input,
        ledger_fact_vendoring,
        chronicle_feed,
        chronicle_epoch,
    )


@cache
def uk_preflight_only_code_hash() -> str:
    """The source hash of the code only the source preflight computes
    through: the shared ledger's registry parity report, behind the ledger
    compile parity gates only the preflight declares
    (``microcosm.build.ledger_targets``).

    The preflight's node carries it as its ``preflight_code`` parameter, so
    the calibrated gates' own key leaves it out (microcosm#931 ruling 6). An
    edit to it re-keys the preflight, the solve after it and the calibrated
    gates through them, and every other node that hashes it: the full build's
    target compilation, selection, measures and problem, the national line's
    targets, problem and gates, and in the in-graph build the
    ``etb_services`` and ``lcfs_consumption`` stages and the transferred
    spine gate, whose own code it is too. Hashed once per process, as the
    calibrated gates' own code is.
    """
    from .. import ledger_targets as shared_ledger_targets

    return source_hash(shared_ledger_targets)


#: Each full gate phase's own code: the hash of what only that phase computes
#: through, which its node carries as its ``<phase>_code`` parameter and the
#: kernel checks before it evaluates anything
#: (:func:`require_current_phase_code`). The code both phases compute through
#: is the gate kernels' shared hash (microcosm#931 ruling 6); test_uk_graph
#: derives each phase's share from the gates it declares and the branch of the
#: kernel's run it takes.
UK_FULL_GATE_PHASE_CODE: Mapping[str, Callable[[], str]] = MappingProxyType(
    {
        "preflight": uk_preflight_only_code_hash,
        "terminal": uk_terminal_only_code_hash,
    }
)


def require_current_phase_code(
    params: Mapping[str, object], phase: str, *, battery: str
) -> None:
    """Refuse a gate node unless its ``<phase>_code`` parameter is its phase's
    own code as this process computes it (microcosm#931 ruling 3).

    ``battery`` is ``"spine"`` or ``"full"``, the battery whose table holds the
    phase's code (:data:`UK_SPINE_GATE_PHASE_CODE`,
    :data:`UK_FULL_GATE_PHASE_CODE`); a phase its table leaves out computes
    through nothing of its own, and its node must carry no parameter. The
    graph construction sets the parameter and node_key hashes it, so a node
    declared in another process, or replayed from a stored graph, with a stale
    or missing parameter is refused before its kernel evaluates anything. The
    check reads only its arguments and the phase's code, and writes nothing,
    so it cannot change what its caller computes: test_uk_graph lists it as
    refusal-only, and no gate kernel's hash binds this module.
    """
    tables = {"spine": UK_SPINE_GATE_PHASE_CODE, "full": UK_FULL_GATE_PHASE_CODE}
    if battery not in tables:
        raise ValueError(f"No UK gate battery {battery!r} carries phase codes.")
    code = tables[battery].get(phase)
    if params.get(f"{phase}_code") != (None if code is None else code()):
        raise ValueError(
            f"UK {battery} gate {phase}-only code differs from its declared binding."
        )
