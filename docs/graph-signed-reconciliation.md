# Signed component reconciliation in the graph

`microcosm.fit.graph_signed_reconciliation.signed_reconciliation_node` exposes
the existing weighted Euclidean projection as an ordinary graph calculation.
Its parameters name the anchor, input draws, output components, nonnegative
bounds, positive scales and numerical tolerances. The country model must
declare and justify those choices. The operator supplies no default scale or
tax interpretation.

For the proposed ACS property-income family, the components are ordinary
interest, retirement-account interest, dividends and signed property income.
The first three have nonnegative bounds; property income can offset them.
A zero reported total therefore does not imply that every component is zero.
Retirement-account interest remains a separate survey component and is not
relabeled as taxable or tax-exempt interest. Source qualification, a defensible
ASEC-to-ACS measurement bridge, fitting and tax conversion remain separate
upstream/downstream operations; this numeric node alone does not complete them.

The graph retains the anchor and raw draw columns. It adds the reconciled
components, each component's adjustment and bound activity, the sum residual
and the projection objective. A typed summary artifact records the complete
rule, row count, adjusted-row count, maximum absolute residual and bound counts.
It describes the numerical result and grants no source or release authority.
These declarations and columns make the rule inspectable in the shared graph
viewer without another viewer implementation.

All input amounts must be finite float64 values. Unknown anchors or draws
refuse; the country host must declare how it selects or models those cases.
Inputs cannot share names with the new outputs, and no original observation
is overwritten. The kernel never reads survey origin, weights or an RNG and
returns no membership or weight changes. Its numerical contract is bitwise on
one platform; the explicit projection tolerances test constraints within each
run and are not a claim of cross-platform equality.

## Verification on 12 September 2026

Twenty-four guarded tests pass, including direct comparison to the independently
tested pure projection; negative and zero-net anchors; exact IDs above 2^53;
row permutation; missing, infinite and incorrectly typed inputs; invalid
bounds/scales and column collisions; and declaration/context disagreement.
An actual two-node graph executes cold, retains the complete input Frame,
metadata and zero-weight record, then reopens the store under required replay.
The replay hits both nodes and preserves artifact identity. Changing scales
reuses the source node and invalidates the reconciliation node.

The closed run used 3.69 seconds wall time and 417,775,616 bytes peak RSS.
All 1,000 source and eight owned hashes, plus 15 resource hashes, were unchanged;
there were no unexpected denied operations or child processes. These are
invented-data graph checks, not native Microcosm acceptance. Receipt SHA-256:
`7e0ed8fefc911b9d7969f48e0d2c382c189e40a0427a120a163a8ce9c8ffae01`.
JUnit SHA-256:
`914745d61c768ae4f445df408db89601d4a4268374ba4f0e6caba6085ea6be6c`.

The first run preserved 22 passing tests and two fixture failures: the test
attempted an empty `Weights` object and constructed unsorted group IDs.
Correcting those fixture inputs produced the result above; production source
was unchanged between the two runs. Independent review is recorded separately.
