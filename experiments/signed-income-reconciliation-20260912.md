# Deterministic reconciliation to a signed income total

`microcosm.fit.signed_reconciliation.reconcile_signed_total` projects a joint
draw onto a caller-qualified total. It is a separate numeric operation for a
future graph node. No existing model, source qualifier, financial attachment,
PUF host or native candidate invokes it in this change.

For draw `q`, anchor `A`, strictly positive scales `s`, and a declared set `B` of
nonnegative components, it solves the strictly convex problem

```text
minimize  sum_j ((z_j - q_j) / s_j)^2
subject to sum_j z_j = A
           z_j >= 0 for j in B
```

At least one component must remain unrestricted, making every finite signed
anchor feasible in exact arithmetic. Component names, the bound mask and scales
are explicit parameters. Ordinary interest and retirement-account interest can
therefore be separate components; this operator assigns neither a source meaning
nor tax treatment to either. It introduces no nonzero-presence restriction.

Define `w_j = (s_j / max(s))^2`. Uniform scale normalization preserves the
minimizer. Given an active bound set `C`, the equality solution for the remaining
free coordinates `F` is

```text
t = (A - sum_{j in F} q_j) / sum_{j in F} w_j
z_j = q_j + t*w_j  for j in F
z_j = 0            for j in C
```

Start with every coordinate free. Fix any bounded coordinate with a negative
candidate to zero, then solve again. Removing negative candidates can only lower
`t`, so a fixed coordinate cannot need releasing later. An unrestricted
coordinate always remains free. The algorithm therefore terminates after at
most `len(B) + 1` solves, without a general optimizer or random draws. The final
free-coordinate stationarity and active-bound dual feasibility are checked.

The signed anchor is never clamped or used as a divisor. A zero total does not
force its components to zero. For example, `[8, 4, -3]` with unit scales and a
zero anchor becomes `[5, 1, -6]`. Positive income and offsetting property loss
remain possible for negative, zero or positive totals. An exactly feasible draw
is preserved, including its float64 signed-zero bits.

Draws have shape `(..., k)` and anchors have exactly the batch shape `(...)`.
Scales and the boolean bound mask each have shape `(k,)`; broadcasting an anchor
across records is deliberately not implicit. Empty batches are supported.
Inputs are finite real numeric values converted into detached float64 arrays.
The result preserves the raw draw, anchor, component roster and scales, together
with projected values, adjustment vectors, signed sum residuals, active bounds,
normalized shifts, objectives and KKT diagnostics. These arrays are descriptive,
mutable values, not an authority or immutable execution receipt.

Callers must supply finite nonnegative `atol` and `rtol`. The sum check uses
`abs(residual) <= atol + rtol*abs(A)`. Component stationarity and dual checks use
the same form with the maximum absolute draw, result and proposed adjustment.
The sum uses `math.fsum`; no residual is patched into a final component and no
display rounding changes the optimum. Nonfinite input, intermediate or objective
overflow, a normalized scale weight lost to underflow, or failure of the stated
tolerances causes refusal. Floating-point refusal is possible even when an
exact-arithmetic solution exists.

The 45-case guarded suite passed with zero failures/errors/skips in 4.470 seconds
wall time, 2.899 CPU seconds and 405,143,552 bytes peak RSS. This includes 160
randomized small problems compared against an independent oracle that enumerates
bound faces, solves dense block KKT systems and selects the minimum objective.
It also verifies negative and zero-net totals, loss offsets, unequal scales,
scale-rescaling invariance, active bounds, exact feasible identity, detached
storage, alternate component rosters, multidimensional and empty batches, invalid
inputs and unstable arithmetic. All 990 source-plus-owned hashes and 15 allowed
resource hashes were unchanged; no child or unexpected refusal occurred. The
normal guard-induced dateutil zoneinfo warning occurred.

Production source SHA256:
`d56c21959121cc7500f06cd747c74bea51561deba537ce2c58266d5966ae84b2`.
Test SHA256: `b5ee23c6ff4f633434c96a9e17c8b82b254435757b6eb24061b1835f57bdaa77`.
The local `codex-signed-reconciliation-20260912/numeric-v2/numeric-v2.json` receipt
SHA256 is `2f071f5116489c1febb4f594c97781c9b8d88d68cf125e94aec446d80e7c3321`.
The earlier guard preflight failure is preserved; it executed no numeric tests.

This numerical acceptance does not establish ACS/ASEC measurement equivalence,
resolve the property or retirement donor bridge, approve scale choices, fit a
model, or certify a release. Source qualification, donor definitions, a named
reconciliation node and dependent model replay remain separate work.
