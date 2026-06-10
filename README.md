# micro

The rebuilt micro stack: one kernel datatype — the **weighted entity bundle** —
and packages as operators on it.

| package | role | succeeds |
|---|---|---|
| `microframe` | the kernel: bundle, typed weights, strata, weighted accounting, unit structure, rules-engine protocol | microdf, microunit |
| `microfit` | conditional models (weight-aware by construction) | microimpute |
| `microcal` | representation: targets → calibrated weights (APG / L0) | microcalibrate |

See [DESIGN.md](DESIGN.md) for the charter: why the rebuild, the kernel
semantics, the RulesEngine protocol (policyengine-us today, Axiom rulespec-us
next), longitudinal design (one weight per trajectory), and the process rules
(behavioral contract tests, constellation versioning, environment-carrying
artifacts).

Legacy consumers (policyengine-us-data, PolicyEngine apps) pin the legacy
packages; there is no backward-compatibility layer.

## Development

```bash
uv sync                  # workspace install
uv run pytest            # all packages, incl. behavioral contract tests
uv run ruff check .
```
