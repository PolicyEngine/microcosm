# US pre-clone housing source graph — progress

Branch `us-housing-source-graph-20260905`, dependent worktree from production
`f01964c7` ("Expose pre-clone prior-year inputs through the US graph").

Authority: `us-housing-source-root-decisions.md` (narrows) over
`us-housing-source-graph-proposal.md` (scope), both under
`/Users/maxghenis/architecture-reviews/microcosm-20260904/`.

## State

Bootstrapping. Isolated locked env created in-worktree with
`env -u UV_FROZEN uv sync --all-packages --locked` (pandas 3.0.3, numpy 2.4.6,
pyarrow 25.0.0, scikit-learn 1.8.0); no shared dependency was modified and no
dependency was upgraded.

## Done

- Read both authority documents; recorded the decisions that bind scope.
- Read the existing seams: `graph_prior_year.py` (closest landed pattern),
  `graph_acs_rent.py`, `graph_acs_rent_source.py`,
  `_graph_acs_rent_artifacts.py`, `graph_context.py`, the registered
  `multispine_pool.py` source helpers, and `housing_inputs.py`.
- Verified the worktree runtime imports from worktree sources.

## Next

1. Capture the tiny direct-original payload/receipt oracle before any edit.
2. Write the failing tests first.
3. Implement `graph_housing_source.py` + artifact codec + registered pool
   helpers.
4. Docs, changelog fragment, inventory, CI grouping, source-blindness.

## Out of scope (explicit)

No source-year money adjustment, no changed rent science or gate thresholds,
no new mandatory post-fit housing signal gate, no shared-merge semantics
change, no real microdata read, no push/deploy.

## Design settled (2026-09-05)

Node topology, all `params={"phase": "prepare_multispine_source_inputs_for_clone"}`:

| Node | Kind | Population / base |
| --- | --- | --- |
| `early_preparation.housing.source` | FILTER, `mass="free"` | base = caller's full pool version |
| `early_preparation.housing.measured` | ordinary | population = `.source` |
| `early_preparation.rent.*` (existing) | unchanged | base = `.source`, context = `.measured` |
| `early_preparation.housing.source_outputs` | ordinary, no cells | population = `early_preparation.rent.boundary` |
| `early_preparation.housing.merge_boundary` | FILTER, `mass="conserve"` | base = caller's full pool version |
| `early_preparation.housing.merge` | ordinary, owns 5 leaves | population = `.merge_boundary` |
| `early_preparation.housing.gate` | GATE | population = `.merge_boundary` |

Representation decisions (from the root decisions doc):

- Source version carries physical `bool` for the two SPM assistance leaves and
  the canonical graph `string` for both tenure leaves; the full-pool merge
  normalizes the booleans to nullable `boolean`, exactly as the direct helper
  already does.
- Full-pool `household.tenure_type` and `spm_unit.spm_unit_tenure_type`
  incumbents are **required** to exist with a `string` representation. Absent
  or incompatible enum representation is refused, not silently widened.
- `fresh_source` is the only mode: selected CPS `pre_subsidy_rent` must be
  absent or entirely null, refused at the boundary before any recipient work.
- The projected operator context has empty metadata and empty mass_log; parent
  authority travels in the separate projection artifact.
