# PUF price baseline — root-review fixes

Bounded fixes to the root review's three actionable findings against the
frozen candidate `8a8f5d23d9d5d20fd85d43cac721dda97e5db5de`. Nothing genuine
runs here either: no PUF delivery, status payload, projection payload,
artifact store or processed dataset is read or statted, no country engine is
imported, nothing is installed and nothing is written to a remote.

## Provenance

| | |
| --- | --- |
| Base | `c1b8324183819cf91511c002bf724a2676524e87` (`origin/main`, fetched 2026-09-06) |
| Candidate under review | `8a8f5d23d9d5d20fd85d43cac721dda97e5db5de` (fast-forward from base) |
| Fixes branch | `us-puf-price-baseline-root-fixes-20260906` |
| Fixes worktree | `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-puf-price-baseline-root-fixes-20260906` |
| Review target worktree | `microcosm-us-puf-price-baseline-20260906`, left frozen and untouched |

## State

Setup complete. Fixes not yet written.

## Done

- Fetched `origin`, confirmed `8a8f5d23` is a descendant of `origin/main`
  `c1b83241`, created this worktree fresh from `origin/main` and fast-forwarded
  it to the candidate. Tree is byte-identical to the candidate's
  (`d0b58af77261ef610acb5521b26edd2a87431161`), `git status --porcelain` empty.
- Read the root review notes, the frozen root plan, the completed preparation
  report, and the four files the findings name.

## Next

1. Close the entry contract in `puf_price_baseline.py`: adapter and invariants
   both validate the incoming `CompiledPufGrowth` against the adopted baseline
   shape, and decline instead of emitting a hardcoded receipt.
2. Cross-check `grown.contract_sha256` and `adaptation.receipt["contract_sha256"]`
   against `compiled.sha256`.
3. Exact kept-row growth comparison in the invariants, using the contract's own
   resolved float64 factor.
4. `design_weight_unchanged` becomes an observed comparison or an honestly
   labelled declaration.
5. `tools/build_us_puf_agi_source_layout.py` restores the cached parent's
   globals, including on failure.
6. Negative tests for each; full regression; regenerate the invented harness.

## Not in scope

Root's fourth note — the primary-resource metadata is caller-asserted and a
genuine runner must verify the frozen BLS bytes and endpoint records — stays
open. This lane does not silently promote caller metadata to an authenticated
public resource, and adds no stat or read of microdata.
