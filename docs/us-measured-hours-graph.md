# Measured hours in the US graph

`us_runtime.graph_early_preparation.us_hours_nodes` declares the existing
pre-clone measured-hours operation as three nodes. Pass the actual upstream
column inventory, population version, and producer of the authenticated
`microcosm.us.frame_context` artifact. Register their implementations with
`register_us_hours_kernels`. Downstream stages must consume `frame_context`
from `early_preparation.hours.gate`.
All three inner nodes declare the authored outer phase
`prepare_multispine_source_inputs_for_clone`; their node IDs remain unchanged.

1. `early_preparation.hours.boundary` validates the existing assembly and
   pre-clone evidence in the registered `multispine_pool` source boundary,
   then creates an all-row graph version.
2. `early_preparation.hours` selects CPS-evidenced people, removes source-role
   columns from the ephemeral person view, runs the actual hours manifest
   handler, computes the original weighted signal observations, and merges
   only `hours_worked_last_week` and `weekly_hours_worked_before_lsr` by person
   ID. It publishes the updated full context and version1 typed measured-hours
   evidence. It does not certify the observations.
3. `early_preparation.hours.gate` applies the original mandatory signal checks
   in their original failure order. It accepts only complete observations
   bound to the same typed producer and exact context payload. On failure,
   the executor saves the gate's original exception message and publishes no
   success context. Both byte and structural descendants remain unreached on
   cold runs, warm runs, and required-cache replay.

The derivation declares every available raw and incumbent hours field. The
executor supplies actual structural IDs/memberships and canonically resolved
person `Weights`; no manually materialized survey-weight column is read. The
registered helper uses the same person-table runner, all-null projection,
ID merge, summary/checks, and ordered receipt builders as the legacy Frame
wrapper. Membership IDs determine the selected group counts exactly as
`Frame.select` prunes each group, including groups without explicit weights.
No partial Frame receives whole-pool metadata. The new graph module has no
source-provenance exemption.

This slice supports one explicit weight source, including household or person
weights. It refuses multiple explicit vectors because the legacy selected
Frame also validates selected sibling weight vectors that the declared person
view cannot inspect. The legacy Frame entrypoint retains its wider API.
Graph-owned incumbent hours leaves must be float64. Missing raw fields are
permitted when all three incumbent hours columns already carry the original
pass-through signal; otherwise the original source runtime refuses them.

All-null union-created hours leaves are absent on the selected source view;
partially present leaves follow the original healing rule. Non-CPS observed
cells remain unchanged. Derived `weeks_worked` is used by the gate and removed
before merging. As in the legacy path, a preexisting full-pool `weeks_worked`
column survives the two-leaf merge; this operation does not clear a stale
formula-owned column or weaken the later terminal guard.

The context preserves untouched entity descriptors, metadata, and the ordered
legacy mass history. Its person column inventory adds only newly owned hours
leaves in the original merge order. The executor retains its own structural
ledger separately. Restoring a legacy checkpoint uses authenticated context;
it must not substitute that history for the graph manifest's ledger.

Hours has its own explicit implementation identity. It binds this graph adapter,
the hours/source-boundary/context helpers, source runtime/manifest and gate
modules, the named source_stages.json resource, and NumPy, pandas and PyArrow
versions. PyArrow can execute string-backed source-evidence operations. Shared
Frame core and graph execution/storage modules are bound conservatively so
changes to projection, dtype/null handling, ownership or replay cannot reuse an
obsolete hours result. The closure does not include source import, assembly,
geography or QRF implementation modules. A change elsewhere inside a listed
whole module also invalidates hours deliberately.

Tests use six synthetic people and the actual manifest handler, with unequal
weights and a nondefault person index. A fixed payload/receipt/context
regression was measured by executing the archived c71e3165 hours and pool
implementations. Presence/healing, native values, weighted and unweighted
group counts, missing source evidence, all-zero selected weight, failure
ordering, forged observations, and replay use the real graph executor. This
is a development slice; it does not activate other preparation stages or
establish terminal pool acceptance.
