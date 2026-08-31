Add explicit, fail-closed Axiom dense-relation bindings and deterministic
relation-batch receipts to the Microcosm frame adapter.  Cross-entity rules now
execute only when callers bind both frame entities and the exact membership
column; the adapter never infers direction from a relation name.
Receipts authenticate owned execution snapshots plus live outputs, preserve
integer or string entity IDs, and fail closed on unsafe repeated native keys.
