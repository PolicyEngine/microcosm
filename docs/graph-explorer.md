# Graph explorer

The graph explorer is one self-contained HTML file generated from a compiled
graph and its run manifest. It contains its own CSS, JavaScript, DAG, and small
charts. A reviewer can copy the file to another machine and open it directly in
a browser; it needs no server, network connection, CDN, font, or charting
library.

## Generate the toy review page

From the repository root, run:

```bash
uv run python tools/graph_demo_run.py --out /tmp/graph-demo
```

Open `/tmp/graph-demo/demo.html`. The directory also contains the canonical
`graph.json`, a normalized `manifest.json`, and the content store needed to
validate and rerender that manifest. The command executes the full toy country,
an inert-description rerun, and all four incident replays. It fails instead of
publishing a misleading page if the inert edit causes a store miss.

The demo is reproducible. Each invocation uses fresh scratch storage and removes
only operational host, clock, and elapsed-time fields from the display manifest.
Two invocations with the same code produce byte-identical HTML.

## Render any saved run

Save both sides of the review contract:

```python
from pathlib import Path

from microcosm.graph import graph_to_json

manifest.save(Path("run/manifest.json"))
Path("run/graph.json").write_text(graph_to_json(graph), encoding="utf-8")
```

By default, put the run's `ContentStore` at `run/store`, beside the manifest.
Then render it with:

```bash
uv run python tools/graph_explain.py \
  --manifest run/manifest.json \
  --graph run/graph.json \
  --out run/explain.html
```

If the store is elsewhere, add `--store /path/to/store`. The renderer uses
`graph_from_json`, compiles the graph again, validates the manifest and every
referenced artifact through `RunManifest.load`, and reloads structural frames
from each receipt's `frame_key`. A missing or corrupt store is an error, not an
unverified page.

For an in-process run, call the pure renderer directly:

```python
from microcosm.graph import explain_html

page = explain_html(
    compiled,
    manifest,
    charter=charter_text,
    burndown=burndown_data,
    replays=replay_data,
)
```

Only `compiled` and `manifest` are required. Omitting the optional inputs leaves
out the acceptance and replay sections. `explain_html` returns a string and does
not read or write files.

## Read the graph explorer

The large SVG is laid out from `CompiledGraph.order` and `predecessors`.
Horizontal position is topological depth. Dashed background groups are the
population versions from `CompiledGraph.versions`, including structural
`create`, `filter`, `expand`, and `reweight` boundaries.

Every node shows its id, kernel reference, role, structural delta, abbreviated
node key, and store hit or miss. Blue fill means a store miss and green fill
means a hit. A gate's border independently records its outcome: passing or
not-applicable gates are green; failed, absent-evidence, or unreached gates are
red. The statuses are always written as text too, so colour is not the only
signal.

Select a node with a pointer, Enter, or Space. Its panel contains:

- the complete node key and the ingredients that enter it;
- every predecessor's full key;
- owned entity/column coordinates, dtypes, row masks, and ownership mode;
- the executor-derived seed;
- the kernel implementation hash and complete capability record;
- column, frame, weight, and opaque artifact identities;
- the complete `NodeReceipt`; and
- the exact run-aware `describe()` view.

A reviewer should first check that the population boundary and predecessor fan-in
match the intended computation. Then check that the selected node owns only the
expected cells, its row mask is narrow enough, and a miss is confined to the
changed node and its descendants.

## Read the acceptance burndown

The demo combines the burndown tool's JSON state with the first explanatory
sentence of every property in `docs/graph-acceptance.md`. The burndown tool owns
A–H; the demo adds V1–V4 as green because this page is their executable review
surface. H2 and H3 remain red until their country fixtures land.

The measured-payoff table is computed, not transcribed. The demo edits only the
chained target's descriptive text, reruns against the populated store, and shows
the exact miss count and share. It also counts the lines in the target's complete
run-aware `describe()` output.

A reviewer should look for a monotone red count and verify that the unrelated
edit reports zero misses. A new red property or a nonzero inert-edit miss is a
regression even when the visual layout still looks plausible.

## Read the calibration view

Every node with a declared `WeightTransition` receives a calibration card; the
view is not keyed to a particular kernel name.

The target table supports both current receipt layouts:

- `calibrate.toy@1` declares `target_entity`, `target_column`, `target_total`,
  and `target_se`; its receipt supplies `achieved` and `mass`.
- `calibrate.adam@1` declares five-field target tuples `(name, measure, filter,
  value, se)`; `receipt.diagnostics.targets[].final_estimate` supplies achieved
  values.

Residual is achieved minus declared target. When a receipt does not expose an
achieved value, residual, standard error, filter, or other field, the table says
“Not recorded” rather than substituting a nearby diagnostic.

The shared-bin histogram compares both incoming and outgoing weights divided by
the original design-weight anchor, aligned by entity id. For a just-executed run,
those arrays come from `manifest.populations`. For a saved run, the CLI reloads
the structural frames identified by `NodeReceipt.frame_key`. A receipt-provided
before/after ratio sample would take precedence if the runtime adds one later.

Mass totals and strata come first from `receipt.mass`, then from the attached
executor `MassRecord`, and finally from the validated before/after structural
frames. The card keeps the declaration's mass policy visible beside the values.

A reviewer should compare each declared target with achieved and residual,
confirm that declared `se` arrived unchanged, scan the histogram for cap pressure
or extreme spread, and check both total and per-stratum mass. “Not recorded” is a
request for more runtime evidence, not a green result.

## Step through incident replays

Each replay has Before, Change, and After buttons. The miniature graph uses the
same predecessor topology as its executed toy fixture. Changed, removed,
rejected, refused, and not-executed nodes receive explicit text and a highlighted
outline. The boxes below each graph list changed nodes, moved keys, cell-boundary
effects, and the observed verdict.

The expected signatures are:

- WIC dtype breach: `wic_recode` is rejected and its reader is unreached.
- `0347a009` repack: five leaves disappear and no survivor key moves.
- Engine-less environment: no graph node or key changes and no kernel executes.
- Evidence flip: no graph key moves and the altered manifest is refused.

All four verdict badges should say `pass`. For the repack and evidence flip, pay
special attention to an empty “Moved node keys” list. For WIC, verify that the
downstream node is not executed after the owning boundary rejects the dtype.

## Portable evidence boundaries

Portable manifest JSON intentionally omits attached populations and transient
mass ledgers. Structural frame artifacts let the CLI recover weights, strata,
and before/after totals, but the current receipt exposes only the realized
maximum weight ratio, not the distribution's samples or bins. The immediate
post-run demo therefore carries richer attached evidence than a manifest copied
without its store. Missing evidence is identified in the page wherever it cannot
be reconstructed faithfully.
