"""Private optional completion custody for the existing atomic financial issuer.

No source or run issuer lives here. Expected receiving/role populations are
constructed once from an actual issued parent; every child borrow checks that
parent and those complete retained values without repeating role source I/O.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import FunctionType

from microcosm.graph import (
    ArtifactInput,
    ArtifactValue,
    KernelRegistry,
    Node,
    StructuralDelta,
)
from microcosm.graph.artifact_edges import numeric_scope
from microcosm.graph.executor import _all_node_keys, _apply_result, _project_context
from microcosm.graph.keys import opaque_artifact_key
from microcosm.graph.population import Population
from microcosm.graph.serialize import graph_to_json

from . import graph_atomic_survey_financial as host
from . import graph_child_property_income as child
from . import graph_survey_completion as receiving

require = host.require


def _roles():
    from . import graph_current_survey_household_roles

    return graph_current_survey_household_roles


def validate_options(options):
    require(type(options) is child.ChildPropertyOptions, "CHILD_OPTIONS_TYPE")
    options.document()  # Explicit scenario, support, transport and keyed stream.


def _live(household_roles):
    result = dict(child._live())
    if household_roles:
        result["roles"] = _roles()._live()
    for module in (sys.modules[__name__], receiving):
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = host.values.source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for method, function in vars(value).items():
                    if isinstance(function, (staticmethod, classmethod)):
                        function = function.__func__
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, method] = (
                            host.values.source._function_seal(function)
                        )
    result["receiving_contract"] = host.values.source._runtime_marker(
        (
            receiving.PROTOCOL,
            receiving.NODE,
            receiving.KERNEL,
            asdict(receiving.CompletionReceivingKernel.capabilities),
        )
    )
    return result


def _source_bytes(household_roles):
    return (
        *child._source_bytes(),
        *(_roles()._source_bytes() if household_roles else ()),
        *(
            (module.__name__, host.codec.sha(Path(module.__file__).read_bytes()))
            for module in (sys.modules[__name__], receiving)
        ),
    )


def _registry(base):
    registry = KernelRegistry()
    for kernel in base.kernels.as_mapping().values():
        registry.register(kernel)
    return registry


def _value(edge, compiled, kernels, keys, loaded):
    """Resolve a descriptor from the compiled producer and verified payload map."""
    node = compiled.graph.node(edge.producer)
    require(
        any(
            o.name == edge.artifact and o.type == edge.type
            for o in node.artifact_outputs
        ),
        "COMPLETION_ARTIFACT_DECLARATION",
    )
    return ArtifactValue(
        loaded[edge.producer, edge.artifact],
        edge.type,
        opaque_artifact_key(keys[edge.producer], edge.artifact),
        keys[edge.producer],
        numeric_scope(kernels.get(node.kernel).capabilities),
    )


def _inputs(node, compiled, kernels, keys, loaded):
    return {
        edge.name: _value(edge, compiled, kernels, keys, loaded)
        for edge in node.artifact_inputs
    }


def _pins(edges, compiled, kernels, keys, loaded):
    result = {}
    for edge in edges:
        value = _value(edge, compiled, kernels, keys, loaded)
        result[edge.name] = dict(
            producer_key=value.producer_key,
            artifact_key=value.key,
            payload_sha256=host.codec.sha(value.payload),
        )
    return result


def _compile(base, nodes):
    return host.compile_graph(
        replace(base.compiled.graph, nodes=(*base.compiled.graph.nodes, *nodes))
    )


def _require_fragment_roster(nodes, identities):
    """Bind each ordered component before it enters any compiled union."""
    require(
        all(type(node) is Node for node in nodes)
        and tuple(node.id for node in nodes) == identities,
        "COMPLETION_FRAGMENT_ROSTER",
    )


def _completion_nodes(boundary, tax_nodes):
    roles = _roles() if boundary.household_roles else None
    fragments = (
        ((boundary.receiving_node,), (receiving.NODE,)),
        (boundary.role_nodes, (roles.SOURCE_NODE, roles.BIND_NODE) if roles else ()),
        (
            boundary.child.nodes,
            (
                child.DONOR,
                child.RECIPIENT,
                child.FIT,
                child.DRAW,
                child.ATTACH,
                child.VERIFY,
            ),
        ),
        (
            tax_nodes,
            (
                host._tax_module().RECEIVING_NODE,
                host._tax_module().TAX_LEAVES_NODE,
                host._tax_module().GATE_NODE,
            ),
        ),
    )
    for nodes, identities in fragments:
        _require_fragment_roster(nodes, identities)
    return tuple(node for nodes, _ in fragments for node in nodes)


def _check_observed_union(base, compiled, manifest, observed):
    require(tuple(observed) == compiled.order, "COMPLETION_OBSERVER_ROSTER")
    require(
        all(manifest.node(n).hit for n in base.compiled.order),
        "COMPLETION_BASE_MUST_HIT",
    )


def _checked_final_io(run, *, source_keys, keys, implementations, loaded):
    """Re-read actual union identities after execution, without running kernels."""
    _, final_source_keys = host._source_paths_and_keys(
        run.compiled, run.sources, run.store
    )
    final_keys, final_implementations = _all_node_keys(
        run.compiled, run.kernels, final_source_keys
    )
    final_loaded = host._artifacts(
        run.manifest,
        run.compiled,
        run.store,
        run.kernels,
        final_keys,
        final_implementations,
    )
    require(
        final_source_keys == source_keys
        and final_keys == keys
        and final_implementations == implementations
        and final_loaded == loaded,
        "COMPLETION_FINAL_IO_CHANGED",
    )
    return final_source_keys, final_loaded


class _CompletionHost:
    """Retained successor values; authority remains with the original base run."""

    def __init__(self, base, options, household_roles):
        self.base = base
        self.base_entry = host._run_entry(base)
        state = self.base_entry[2]
        require(
            state.property_income is not None and not state.rebase_property_taxes,
            "COMPLETION_ACTUAL_PROPERTY_PARENT",
        )
        self.options, self.household_roles = options, household_roles
        self.options_bytes = child._json(options.document())
        self.revoked = False
        self.live = _live(household_roles)
        self.source_bytes = _source_bytes(household_roles)
        self.expected, self.receipts, self.payloads = {}, {}, {}
        self.role_qualified = None
        self.role_nodes = ()
        self.child = None
        self.completed = None
        self.observed = None
        self.observed_stamps = None
        self.final_states = None
        self.anchor = None
        self.base.checked_view()
        self.kernels = _registry(base)
        self.source_keys = dict(state.source_keys)
        base_keys = dict(state.keys)
        loaded = host._artifacts(
            base.manifest,
            base.compiled,
            base.store,
            base.kernels,
            base_keys,
            dict(state.implementations),
        )
        property_graph = host._property_module()
        reconciliation = state.property_income.nodes(
            (property_graph.PROPERTY_REPORTED_TOTAL,)
        )[-1]
        actual = base.compiled.graph.node(reconciliation.id)
        output = actual.artifact_outputs[0]
        self.ordering = (
            ArtifactInput(
                "property_projection",
                property_graph.PROJECTION_NODE,
                "projection",
                property_graph.PROJECTION_TYPE,
            ),
            ArtifactInput(
                "property_reconciliation", actual.id, output.name, output.type
            ),
        )
        self.host_edges = (
            *host.financial.host.current_survey_host_edges(),
            host.financial._geography_edge(),
            *self.ordering,
        )
        self.host_pins = _pins(
            self.host_edges, base.compiled, base.kernels, base_keys, loaded
        )
        parent = base.financial_population
        self.receiving_node = receiving.completion_receiving_node(
            parent.frame, population=parent.version, ordering=self.ordering
        )
        _require_fragment_roster((self.receiving_node,), (receiving.NODE,))
        self.kernels.register(receiving.CompletionReceivingKernel())
        compiled = _compile(base, (self.receiving_node,))
        keys, _ = _all_node_keys(compiled, self.kernels, self.source_keys)
        context = _project_context(
            self.receiving_node,
            parent,
            key=keys[receiving.NODE],
            sources={},
            tolerances={},
            numerics={},
            artifacts=_inputs(
                self.receiving_node, compiled, self.kernels, keys, loaded
            ),
        )
        result = receiving.CompletionReceivingKernel().run(context)
        self.receiving = _apply_result(
            self.receiving_node,
            result,
            parent,
            mass_partition=compiled.graph.mass_partition,
        )
        self.expected[receiving.NODE] = self.receiving
        self.receipts[receiving.NODE] = result.receipt
        self.parent = self.receiving
        if household_roles:
            roles = _roles()
            self.role_qualified = roles.qualify_current_survey_household_roles(
                base.prefix.preparation
            )
            self.reconciliation = roles.roles.household_role_reconciliation(
                self.role_qualified, self.receiving.frame
            )
            require(
                self.reconciliation["conflicting_cells"] == 0
                and self.reconciliation["incumbent_known_qualified_unbound"] == 0,
                "HOUSEHOLD_ROLE_RECONCILIATION",
            )
            role_edges = roles.country_host_edges()
            role_pins = _pins(
                role_edges, base.compiled, base.kernels, base_keys, loaded
            )

            def require_current():
                self.pure()

            self.role_nodes = roles.register_household_role_kernels(
                self.kernels,
                self.role_qualified,
                self.receiving.frame,
                receiving_version=self.receiving.version,
                host_pins=role_pins,
                after=self.ordering[-1],
                host_edges=role_edges,
                require_current=require_current,
            )
            _require_fragment_roster(
                self.role_nodes, (roles.SOURCE_NODE, roles.BIND_NODE)
            )
            compiled = _compile(base, (self.receiving_node, *self.role_nodes))
            keys, _ = _all_node_keys(compiled, self.kernels, self.source_keys)
            expected_payloads = dict(loaded)
            for node in self.role_nodes:
                incoming = (
                    None
                    if node.structural is StructuralDelta.CREATE
                    else self.receiving
                )
                result = roles._result(
                    self.role_qualified,
                    node,
                    None if incoming is None else incoming.frame.person,
                )
                population = roles.expected_survey_household_roles_population(
                    node.id,
                    incoming,
                    qualified=self.role_qualified,
                    node=node,
                    artifacts=_inputs(
                        node, compiled, self.kernels, keys, expected_payloads
                    ),
                )
                self.expected[node.id], self.receipts[node.id] = (
                    population,
                    result.receipt,
                )
                for name, payload in result.artifacts.items():
                    self.payloads[node.id, name] = expected_payloads[node.id, name] = (
                        payload
                    )
            self.parent = self.expected[roles.BIND_NODE]
            roles.verify_materialized_survey_household_roles(
                self.parent,
                self.receiving,
                qualified=self.role_qualified,
                node=self.role_nodes[-1],
            )
        self.anchor = self._seal()

        def require_parent():
            self.pure()
            return self.parent

        self.child = child.ChildPropertyBoundary(
            base.prefix.preparation,
            self.parent,
            require_parent=require_parent,
            options=options,
            host_edges=self.host_edges,
            host_pins=self.host_pins,
        )
        self.anchor = self._seal()
        for kernel in self.child.kernels():
            self.kernels.register(kernel)
        self.requalify()

    def _seal(self):
        return host.values.source._runtime_marker(
            (
                id(self.base),
                id(self.base_entry),
                id(self.kernels),
                id(self.child),
                self.household_roles,
                self.options_bytes,
                self.options.document(),
                self.source_bytes,
                self.host_pins,
                tuple(asdict(edge) for edge in self.host_edges),
                self.receiving_node.normative(),
                tuple(n.normative() for n in self.role_nodes),
                tuple(
                    (n, host.reconstruction._population_stamp(p))
                    for n, p in self.expected.items()
                ),
                host.reconstruction._population_stamp(self.parent),
                None
                if self.role_qualified is None
                else _roles().household_role_projection_seal(self.role_qualified),
            )
        )

    def attestation(self):
        """Issuer-retained identity fence over every actually observed object."""
        return (
            self.anchor,
            self._seal(),
            id(self.receiving),
            id(self.parent),
            id(self.role_qualified),
            id(self.child),
            id(self.observed),
            tuple((name, id(population)) for name, population in self.observed.items()),
            self.observed_stamps,
        )

    def revoke(self):
        self.revoked = True
        if self.child is not None:
            self.child.revoked = True

    def pure(self):
        try:
            require(not self.revoked, "COMPLETION_HOST_REVOKED")
            host._pure_run(self.base, self.base_entry)
            require(_live(self.household_roles) == self.live, "COMPLETION_LIVE_CHANGED")
            if self.anchor is not None:
                require(self._seal() == self.anchor, "COMPLETION_RETAINED_CHANGED")
            if self.child is not None:
                self.child._pure()
            if self.observed is not None:
                require(
                    tuple(
                        (n, host.reconstruction._population_stamp(p))
                        for n, p in self.observed.items()
                    )
                    == self.observed_stamps,
                    "COMPLETION_OBSERVED_CHANGED",
                )
                require(
                    self.completed is self.observed[child.VERIFY],
                    "COMPLETION_OUTPUT_IDENTITY",
                )
        except Exception:
            self.revoke()
            raise

    def requalify(self):
        """Full original-source/role checks at host boundaries, never per borrow."""
        self.pure()
        self.base.checked_view()
        if self.role_qualified is not None:
            roles = _roles()
            current = roles.qualify_current_survey_household_roles(
                self.base.prefix.preparation
            )
            require(
                roles.household_role_projection_seal(current)
                == roles.household_role_projection_seal(self.role_qualified),
                "COMPLETION_ROLE_SOURCE_CHANGED",
            )
            require(
                roles.roles.household_role_reconciliation(current, self.receiving.frame)
                == self.reconciliation,
                "COMPLETION_ROLE_RECONCILIATION_CHANGED",
            )
            roles.verify_materialized_survey_household_roles(
                self.parent, self.receiving, qualified=current, node=self.role_nodes[-1]
            )
        require(
            _source_bytes(self.household_roles) == self.source_bytes,
            "COMPLETION_SOURCE_CHANGED",
        )
        self.pure()

    def tax_edge(self):
        return ArtifactInput(
            "child_verification", child.VERIFY, "verification", child.VERIFICATION_TYPE
        )

    def owned_columns(self):
        return (
            child.STATUS,
            child.IMPUTED,
            *((_roles().roles.CANONICAL_COLUMN,) if self.household_roles else ()),
        )

    def document(self):
        # _child_expected reconstructs these bytes; verify() compares them to
        # actual store evidence through the real child verifier before issuance.
        verification = host.codec.decode_json(
            self.payloads[child.VERIFY, "verification"]
        )
        return {
            "household_roles": self.household_roles,
            "child_property": self.options.document(),
            "completion_receiving_node_count": 1,
            "household_role_node_count": len(self.role_nodes),
            "child_property_node_count": len(self.child.nodes),
            "property_completion_routing_scope": "pre_child_source_gaps",
            "completion_parent_sha256": host.codec.sha(self.base_entry[1]),
            "child_verification_sha256": host.codec.sha(
                self.payloads[child.VERIFY, "verification"]
            ),
            "child_source_knownness_changed": verification["source_knownness_changed"],
        }

    def verify(self, run, *, keys, loaded):
        """The final verifier always receives actual executor/store evidence."""
        try:
            evidence = tuple(sorted(loaded.items()))
            self.requalify()
            require(
                self.observed is not None, "COMPLETION_ACTUAL_OBSERVATIONS_REQUIRED"
            )
            self.child.verify_materialized(
                self.observed[child.VERIFY],
                _inputs(
                    run.compiled.graph.node(child.VERIFY),
                    run.compiled,
                    run.kernels,
                    keys,
                    loaded,
                ),
                support_populations={
                    n: self.observed[n] for n in (child.DONOR, child.RECIPIENT)
                },
                verification=_value(
                    self.tax_edge(), run.compiled, run.kernels, keys, loaded
                ),
            )
            for node_id, expected in self.expected.items():
                host.atomic.same_replayed_population(expected, self.observed[node_id])
            require(
                all(loaded[key] == payload for key, payload in self.payloads.items()),
                "COMPLETION_MATERIALIZED_ARTIFACT",
            )
            host.survey._check_node_states(run.manifest, self.final_states)
            self.requalify()
            require(
                tuple(sorted(loaded.items())) == evidence,
                "COMPLETION_FINAL_EVIDENCE_CHANGED",
            )
            self.pure()
        except Exception:
            self.revoke()
            raise


def _child_expected(boundary):
    child_boundary = boundary.child
    completed, result, payloads, donor, recipient = child._reconstruct(
        child_boundary.qualified,
        child_boundary.origins,
        boundary.parent,
        boundary.options,
        child_boundary.nodes,
    )
    supports = {
        name: Population.from_frame(child._support_frame(table), name)
        for name, table in ((child.DONOR, donor), (child.RECIPIENT, recipient))
    }
    expected = {
        **supports,
        child.FIT: supports[child.DONOR],
        child.DRAW: supports[child.RECIPIENT],
        child.ATTACH: completed,
        child.VERIFY: completed,
    }
    draw = next(n for n in child_boundary.nodes if n.id == child.DRAW)
    p = draw.params
    receipts = {
        **{
            name: {"private_support_rows": len(table), "source_admission_issued": False}
            for name, table in ((child.DONOR, donor), (child.RECIPIENT, recipient))
        },
        child.FIT: host.codec.decode_json(payloads[child.FIT, "model_metadata"]),
        child.DRAW: dict(
            protocol=child.adapter.PROTOCOL,
            rows=int(recipient.eligible.sum()),
            candidate_rows=len(recipient),
            model_sha256=host.codec.sha(payloads[child.FIT, "model"]),
            **{
                name: p[name]
                for name in (
                    "support_sha256",
                    "donor_source_sha256",
                    "source_sha256",
                    "scenario_sha256",
                    "transport_sha256",
                )
            },
            draw_sha256=host.codec.sha(payloads[child.DRAW, "draws"]),
        ),
        child.ATTACH: result.receipt,
        child.VERIFY: dict(
            original_recipient_count=int(recipient.eligible.sum()),
            source_observation_claim=False,
            full_population_owner_check_required=True,
        ),
    }
    return completed, expected, receipts, payloads


def extend(base, *, options, household_roles, resume):
    """Execute the actual union in the same store, then use the existing issuer."""
    validate_options(options)
    boundary = _CompletionHost(base, options, household_roles)
    try:
        return _extend(boundary, resume=resume)
    except Exception:
        boundary.revoke()
        raise


def _extend(boundary, *, resume):
    base, state = boundary.base, boundary.base_entry[2]
    expected_completed, expected_child, child_receipts, child_payloads = (
        _child_expected(boundary)
    )
    # The declaration uses the independently completed real Frame, including all
    # status/role/imputation columns. It never guesses anticipated child writes.
    tax_nodes = host._tax_nodes(
        expected_completed.frame,
        expected_completed.version,
        state.property_income,
        completion=boundary.tax_edge(),
    )
    nodes = _completion_nodes(boundary, tax_nodes)
    compiled = _compile(base, nodes)
    require(
        len(compiled.order) == len(base.compiled.order) + len(nodes),
        "COMPLETION_UNION_ROSTER",
    )
    tax = host._tax_module()
    for cls in (
        tax.PropertyTaxReceivingKernel,
        tax.PropertyTaxLeavesKernel,
        tax.PropertyTaxLeafGateKernel,
    ):
        boundary.kernels.register(cls())
    kernels = boundary.kernels
    keys, implementations = _all_node_keys(compiled, kernels, boundary.source_keys)
    require(all(keys[n] == k for n, k in state.keys), "COMPLETION_BASE_KEYS_CHANGED")
    declaration = graph_to_json(compiled.graph)
    live = host._live(
        state.property_income, True, state.person_status_boundary is not None, boundary
    )
    observed, stamps = {}, {}

    def observe(node_id, population):
        require(node_id not in observed, "COMPLETION_OBSERVER_DUPLICATE")
        observed[node_id] = population
        stamps[node_id] = host.reconstruction._population_stamp(population)

    manifest = host.run_graph(
        compiled,
        sources=base.sources,
        store=base.store,
        kernels=kernels,
        resume=resume,
        _population_observer=observe,
    )
    _check_observed_union(base, compiled, manifest, observed)
    loaded = host._artifacts(
        manifest, compiled, base.store, kernels, keys, implementations
    )
    require(
        all(
            host.codec.sha(loaded[n, a]) == digest
            for n, a, digest in state.artifact_hashes
        ),
        "COMPLETION_BASE_ARTIFACT_CHANGED",
    )
    tax_populations, tax_results = host._reconstruct_tax(
        observed[child.VERIFY],
        compiled=compiled,
        kernels=kernels,
        keys=keys,
        loaded=loaded,
        options=state.property_income,
        completion=boundary.tax_edge(),
    )
    base_expected = dict(
        zip(base.compiled.order, (p for p, _ in state.node_populations), strict=True)
    )
    expected = {
        **base_expected,
        **boundary.expected,
        **expected_child,
        **tax_populations,
    }
    receipts = {
        **{n: r.receipt for n, r in base.manifest.nodes.items()},
        **boundary.receipts,
        **child_receipts,
        **{n: result.receipt for n, result in tax_results.items()},
    }
    for node_id in compiled.order:
        host.atomic.same_replayed_population(expected[node_id], observed[node_id])
    states = host.atomic._states(
        compiled, kernels, boundary.source_keys, expected, receipts
    )
    host.survey._check_node_states(manifest, states)
    boundary.payloads.update(child_payloads)
    boundary.completed = observed[child.VERIFY]
    boundary.observed = observed
    boundary.observed_stamps = tuple(stamps.items())
    boundary.final_states = states
    result = host.AtomicSurveyFinancialRunValues(
        base.prefix,
        observed[tax.GATE_NODE],
        manifest,
        compiled,
        base.store,
        kernels,
        dict(base.sources),
        base.projection,
        base.matrix,
    )
    manifest_bytes = manifest.to_json_bytes()
    manifest_populations = host._manifest_population_seals(manifest, compiled)
    registry = tuple(kernels.as_mapping().items())
    # Last implementation/source/store I/O comes before all final pure seals.
    source_keys, final_loaded = _checked_final_io(
        result,
        source_keys=boundary.source_keys,
        keys=keys,
        implementations=implementations,
        loaded=loaded,
    )
    boundary.verify(result, keys=keys, loaded=final_loaded)
    boundary.pure()
    require(
        manifest.to_json_bytes() == manifest_bytes
        and host._manifest_population_seals(manifest, compiled) == manifest_populations
        and tuple(kernels.as_mapping().items()) == registry,
        "COMPLETION_FINAL_MANIFEST_CHANGED",
    )
    require(
        graph_to_json(compiled.graph) == declaration
        and compiled == host.compile_graph(compiled.graph)
        and host._live(
            state.property_income,
            True,
            state.person_status_boundary is not None,
            boundary,
        )
        == live,
        "COMPLETION_FINAL_DECLARATION_CHANGED",
    )
    for node_id, population in expected.items():
        host.atomic.same_replayed_population(population, observed[node_id])
    host.survey._check_node_states(manifest, states)
    host._issue_run(
        result,
        preparation_entry=state.preparation_entry,
        pins=host.codec.decode_json(state.pins),
        n_estimators=state.n_estimators,
        demographic_conditioning=state.demographic_conditioning,
        source_keys=source_keys,
        keys=keys,
        implementations=implementations,
        loaded=loaded,
        live=live,
        property_income=state.property_income,
        legacy_financial_population=state.legacy_financial_population,
        rebase_property_taxes=True,
        property_population=base.financial_population,
        person_status_boundary=state.person_status_boundary,
        completion_boundary=boundary,
        node_populations=observed,
        node_states=states,
    )
    return result
