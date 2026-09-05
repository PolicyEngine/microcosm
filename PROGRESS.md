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
