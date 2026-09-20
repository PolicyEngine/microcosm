"""Original PUF application owned by the existing survey-enrichment host.

This is a specific internal binding, not another run issuer. The existing host
admits ancestors, reconstructs every prior enrichment branch and issues only
its own completed result. The executor's terminal observation supplies the full
receiving population; declaration templates never provide receiving authority.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Numeric,
    StructuralDelta,
    executor,
    source_hash,
)
from microcosm.graph import population as population_ops
from microcosm.graph.serialize import _node_payload

from . import graph_puf55_original_placement as fragment

application, values = fragment.application, fragment.values
fixed_graph = application.fixed_graph
recipient_graph = fixed_graph.parent
codec, physical = values.codec, values.attachment.physical
require = values.require


def modules():
    return (
        sys.modules[__name__],
        fragment,
        values,
        application,
        fixed_graph,
        fixed_graph.values,
        recipient_graph,
        recipient_graph.values,
        fixed_graph.values.routing,
        fixed_graph.values.leaves,
        application.observed,
        population_ops,
        executor,
    )


def configuration():
    return (
        values.PROTOCOL,
        values.SINGLETON_OUTPUTS,
        values.UNIT_OUTPUTS,
        fixed_graph.values.DEVELOPMENT_RULES,
        fixed_graph.values.FINANCIAL_TARGETS,
        fixed_graph.values.DEVELOPMENT_TARGETS,
        fragment.KEEP_NODE,
        fragment.ATTACH_NODE,
        fragment.KEEP_REF,
        fragment.ATTACH_REF,
        fragment.PLACEMENT_TYPE,
    )


class Binding:
    """Host-owned retained source/terminal state; never an independently issued run."""

    def __init__(self, host, *, terminal, application_seed):
        self.host = host
        self.terminal = terminal
        puf_boundary = host.parent_entry[2].boundary
        application._seeds(puf_boundary.seed, application_seed)
        self.seeds = dict(
            clone_one_seed=puf_boundary.seed, original_application_seed=application_seed
        )
        self.routes = puf_boundary.routes
        self.fixed = fixed_graph.values.qualify_puf55_survey_fixed_inputs(
            host.run.financial_run,
            arm=0,
            development_rules=fixed_graph.values.DEVELOPMENT_RULES,
        )
        self.fixed_stamp = fixed_graph.values.fixed_input_stamp(self.fixed)
        self.template = values._capture_checked_puf_ancestors(
            host.run, host.run.population
        )
        self.template_stamp = values._stamp(self.template)
        self.source_nodes = (
            *recipient_graph.puf55_survey_recipient_nodes(self.fixed.recipients),
            *fixed_graph.puf55_survey_fixed_input_nodes(self.fixed),
        )
        self.apply_nodes = application.original_application_nodes(
            self.fixed, self.routes, population=terminal.population, **self.seeds
        )
        after = host._original_after(terminal)
        self.after = after
        self.placement_nodes = fragment.original_placement_nodes(
            self.fixed,
            self.template,
            self.routes,
            after=after,
            receiving_version=terminal.population,
            **self.seeds,
        )
        self.nodes = (*self.source_nodes, *self.apply_nodes, *self.placement_nodes)
        self.node_stamp = tuple(codec.encode_json(_node_payload(n)) for n in self.nodes)
        self.config = (
            configuration(),
            tuple(sorted(self.seeds.items())),
            self.routes,
            terminal,
            after,
        )
        self.receiving = self.receiving_stamp = self.kept = self.kept_stamp = None

    def pure(self):
        require(
            self.config
            == (
                configuration(),
                tuple(sorted(self.seeds.items())),
                self.routes,
                self.terminal,
                self.after,
            )
            and self.routes == self.host.parent_entry[2].boundary.routes
            and self.seeds["clone_one_seed"] == self.host.parent_entry[2].boundary.seed
            and values._stamp(self.template) == self.template_stamp
            and fixed_graph.values.fixed_input_stamp(self.fixed) == self.fixed_stamp
            and tuple(codec.encode_json(_node_payload(n)) for n in self.nodes)
            == self.node_stamp,
            "HOST_BINDING_CHANGED",
        )
        require(self.terminal in self.host.declaration, "TERMINAL_DECLARATION")
        if self.receiving is None:
            require(
                self.receiving_stamp is self.kept is self.kept_stamp is None,
                "TERMINAL_UNOBSERVED",
            )
        else:
            require(
                physical._population_stamp(self.receiving) == self.receiving_stamp
                and physical._population_stamp(self.kept) == self.kept_stamp,
                "TERMINAL_CHANGED",
            )

    def observe_terminal(self, node_id, population):
        if node_id != self.terminal.id:
            return
        self.host.pure()
        require(
            self.receiving is None and population.version == self.terminal.population,
            "TERMINAL_OBSERVER",
        )
        inputs = replace(self.template, receiving=population)
        values._axes(inputs, self.fixed)
        expected = fragment.original_placement_nodes(
            self.fixed, inputs, self.routes, after=self.after, **self.seeds
        )
        require(expected == self.placement_nodes, "TERMINAL_OUTPUT_DECLARATIONS")
        self.receiving = population
        self.receiving_stamp = physical._population_stamp(population)
        self.kept = fragment.keep_all_population(inputs, expected[0])
        self.kept_stamp = physical._population_stamp(self.kept)
        self.host.pure()

    def inputs(self):
        self.pure()
        require(self.receiving is not None, "TERMINAL_NOT_OBSERVED")
        return replace(self.template, receiving=self.receiving)

    def context(self, context):
        self.host.context(context)
        require(context.node in self.placement_nodes, "PLACEMENT_CONTEXT_NODE")
        expected = (
            self.inputs().receiving
            if context.node.id == fragment.KEEP_NODE
            else self.kept
        )
        require(
            context.node.base == expected.version
            if context.node.structural is StructuralDelta.FILTER
            else context.node.population == expected.version,
            "PLACEMENT_CONTEXT_VERSION",
        )
        # Compare every supplied declared table against the actual complete
        # terminal/keep-all Population retained by the executor observation.
        projected = executor._project_context(
            context.node,
            expected,
            key=dict(self.host.keys)[context.node.id],
            sources=dict(self.host.paths),
            tolerances=context.tolerances,
            numerics=context.numerics,
            artifacts=context.artifacts,
        )
        require(
            context.params == context.node.params
            and executor._context_digest(context)
            == executor._context_digest(projected),
            "PLACEMENT_CONTEXT_VALUES",
        )
        self.pure()

    def verify_sources(self, manifest, loaded):
        projection, matrix = recipient_graph.recipient_node_ids(0)
        require(
            loaded[projection, "projection"] == self.fixed.recipients.receipt,
            "ORIGINAL_PROJECTION",
        )
        matrices = {}
        for profile, expected in self.fixed.recipients.matrices:
            name = recipient_graph._NAMES[profile]
            require(loaded[matrix, name] == expected, "ORIGINAL_MATRIX")
            matrices[profile] = expected, manifest.node(matrix).key
        payloads = fixed_graph._payloads(self.fixed, matrices)
        fixed = fixed_graph.fixed_input_node_id(0)
        require(
            all(loaded[fixed, name] == payload for name, payload in payloads.items()),
            "ORIGINAL_FIXED_ARTIFACTS",
        )
        self.pure()

    def requalify(self):
        fresh = fixed_graph.values.qualify_puf55_survey_fixed_inputs(
            self.host.run.financial_run,
            arm=0,
            development_rules=fixed_graph.values.DEVELOPMENT_RULES,
        )
        require(
            fixed_graph.values.fixed_input_stamp(fresh) == self.fixed_stamp,
            "ORIGINAL_SOURCE_REQUALIFICATION",
        )
        self.pure()

    def reconstruct(self, node, incoming, artifacts, persisted):
        """Called in the host's independent ordered reconstruction, not its observer."""
        self.pure()
        inputs = self.inputs()
        if node.id == fragment.KEEP_NODE:
            physical.replay.same_replayed_population(incoming, inputs.receiving)
            expected = fragment.keep_all_population(
                replace(inputs, receiving=incoming), node
            )
            require(not persisted, "KEEP_ARTIFACTS")
        else:
            require(node.id == fragment.ATTACH_NODE, "RECONSTRUCTION_NODE")
            physical.replay.same_replayed_population(incoming, self.kept)
            result = fragment.original_placement_result(
                node,
                self.fixed,
                inputs,
                self.routes,
                artifacts,
                after=self.after,
                **self.seeds,
            )
            require(persisted == result.artifacts, "PLACEMENT_ARTIFACTS")
            expected = population_ops.patch(incoming, node, result)
        self.pure()
        return expected


class _Kernel(KernelBase):
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=recipient_graph._Kernel.capabilities.dependencies,
    )

    def __init__(self, binding):
        self.binding = binding

    def implementation_hash(self):
        # Existing host kernels and ancestor sources retain their own closures.
        return source_hash(
            *modules(),
            sys.modules[type(self.binding.host).__module__],
            _node_payload,
            dependencies=self.capabilities.dependencies,
        )


class KeepAllKernel(_Kernel):
    ref = fragment.KEEP_REF
    capabilities = replace(_Kernel.capabilities, structural=StructuralDelta.FILTER)

    def run(self, context):
        b = self.binding
        b.context(context)
        ids = b.inputs().receiving.frame.person.person_id
        result = KernelResult(
            keep=pd.Series(
                np.ones(len(ids), dtype=bool), index=pd.Index(ids, name="person_id")
            )
        )
        b.context(context)
        return result


class PlacementKernel(_Kernel):
    ref = fragment.ATTACH_REF

    def run(self, context):
        b = self.binding
        b.context(context)
        result = fragment.original_placement_result(
            context.node,
            b.fixed,
            b.inputs(),
            b.routes,
            context.artifacts,
            after=b.after,
            **b.seeds,
        )
        seal = (
            codec.encode_json(result.receipt),
            tuple(sorted(result.artifacts.items())),
            tuple(
                (key, values.values.recipients._table_digest(column.to_frame()))
                for key, column in result.columns.items()
            ),
        )
        b.context(context)
        require(
            seal
            == (
                codec.encode_json(result.receipt),
                tuple(sorted(result.artifacts.items())),
                tuple(
                    (key, values.values.recipients._table_digest(column.to_frame()))
                    for key, column in result.columns.items()
                ),
            ),
            "FINAL_PLACEMENT_RESULT",
        )
        return result


def kernels(binding):
    return (
        fixed_graph.Puf55SurveyFixedInputKernel(
            binding.host.run.financial_run,
            development_rules=fixed_graph.values.DEVELOPMENT_RULES,
        ),
        application.observed.LegacyQRFApplyObservedMatrixKernel(),
        KeepAllKernel(binding),
        PlacementKernel(binding),
    )
