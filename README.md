# populace

The rebuilt micro stack: one kernel datatype — the **`Frame`**, a weighted
sampling frame of entity tables — and packages as operators on it. One PEP 420
`populace` namespace, shipped as shard distributions; a `populace` metapackage
will pin the constellation.

| package | import | role | succeeds |
|---|---|---|---|
| `populace-frame` | `populace.frame` | the kernel: Frame, typed weights, strata, links, weighted accounting, unit structure, rules-engine protocol | microdf, microunit |
| `populace-fit` | `populace.fit` | conditional models (weight-aware by construction) | microimpute |
| `populace-calibrate` | `populace.calibrate` | representation: targets → calibrated weights (APG / L0) | microcalibrate |

`microplex` — the engine — keeps its own repo and brand, and re-bases onto
populace-frame stage by stage.

See [DESIGN.md](DESIGN.md) for the charter: why the rebuild, the kernel
semantics, the RulesEngine protocol (policyengine-us today, Axiom rulespec-us
next), longitudinal design (one weight per trajectory), and the process rules
(behavioral contract tests, constellation versioning, environment-carrying
artifacts).

Legacy consumers (policyengine-us-data, PolicyEngine apps) pin the legacy
packages; there is no backward-compatibility layer.

## Development

```bash
uv sync --all-packages   # workspace install (all members + dev groups)
uv run pytest            # all packages, incl. behavioral contract tests
uv run ruff check .
```
