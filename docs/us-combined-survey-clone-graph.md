# US combined-survey PUF-support clone graph stage

`microcosm.build.us_runtime.graph_combined_clone` turns an already combined
ASEC + ACS population into the two-role host the PUF detail pass needs. It is
the whole-household clone stage as a real graph `EXPAND`, and it is only that.

## What it does

The input is a combined native host: every entity carries the four support
provenance columns, every `*_support_clone_index` is zero, both survey source
channels are present, and household weights are already `IMPORTANCE`. The
output adds exactly one whole-household PUF-detail clone of every household —
every person, tax unit, SPM unit, family and marital unit — and splits each
incoming household weight evenly across the native/detail pair.

Both source channels are carried unchanged into both roles. `asec` and
`puf_tax_detail` are *operator roles* on the clone-index axis; they are not
source channels and they never overwrite `person_support_channel` and its
siblings. The stratum column, `*_source_id`, `*_spine_source_id` and every
amount are carried by the executor from the copied rows.

## What it deliberately does not do

No PUF donor is read, no quantile forest is fitted, no draw is taken, no amount
is placed, no tax or benefit value is computed, and no prior-wage stage runs.
The clone node declares no source and no typed artifact input, so no donor
artifact can reach it; the donor sampling weight `S006` never enters a host
weight because this stage never sees a donor. Partial (seeded fraction)
attachment is a separate explicit control and is not reachable here: the stage
pins `clone_attachment_fraction=1.0` and `clone_attachment_seed=0` and refuses
any other value.

The profile is `development`; the receipt records `release_eligible: false`.
Nothing here is a calibration, a certified population or a release.

## Why an `EXPAND`, and what the kernel returns

`clone_us_frame_for_puf_support` already performs this operation directly and
remains the behavioral authority: the kernel calls it and validates its output
with `validate_puf_clone_attachment`, exactly as the direct path does. What the
kernel hands the executor, though, is not that Frame. It returns what
`KernelResult` declares for a structural node — per-entity clone lineage, the
six declared `*_support_clone_index` overlays, and explicit household weights —
and `microcosm.graph.population._patch_expand` does the structural work: it
carries every column from each copied row, remaps every copied person's five
memberships onto the copied groups, appends the strata, installs the weights
and records the mass ledger. Handing the executor an opaque Frame would make
its lineage, membership, storage, weight and mass checks vacuous.

A copied row's clone index differs from its source's, which the executor
accepts only when a same-version claimant declares that coordinate a full-cell
rewrite. `us_combined_survey_clone_nodes` therefore returns two nodes: the
`EXPAND` and a claim node whose population is the `EXPAND` and whose six
outputs are `rewrite=True`. The claim kernel re-emits exactly the values the
`EXPAND` installed.

## Identity: the declared key, not the integers

The executor's contract is that every base row keeps its own id and new rows
get new ids. On this input the direct operator honours the same rule — for a
preassembled frame it copies the native block unchanged and shifts only the
detail block by a decimal multiplier — so the two paths do in fact agree on raw
ids. The stage does not depend on that. Lineage is derived from the operator's
own output through the `(entity_source_id, clone_index)` bijection, with the
positional pairing checked independently; a disagreement between the two is a
refusal (`LINEAGE_POSITION_DISAGREES`), not a silent choice between them. The
tests compare graph against direct through that declared key and state the raw
id agreement as a separate, observed fact.

## Mass

The node declares `mass="conserve"`. Halving household weights while doubling
rows leaves the executor's weighted person mass per stratum unchanged, so the
ledger is a real check rather than a declared one. Each pair's weights sum to
their parent bit for bit: halving a finite double and adding the two halves is
exact, so the kernel requires equality rather than a tolerance.

## What the node key binds

The kernel's implementation hash covers this module, `puf_support` and
`support_provenance`, plus the pinned `numpy`/`pandas` versions, so an edit to
the clone operator moves every cached identity. The node key additionally binds
its declared parameters, the artifact keys of the twenty-four provenance
columns it reads and — through `population_input['base']` — the whole base
population version: its amounts, ids, row order, provenance and incoming
weights. A column the kernel never reads still enters the key that way, which
the test suite demonstrates by perturbing an amount, an id, the row order and
an incoming weight and observing four different node keys.

## The assembly manifest

`validate_assembly_provenance` needs `Frame.metadata`, and `KernelContext` has
no metadata channel: the content store persists tables, weights, strata and the
mass log, not `Frame.metadata`. The kernel therefore rebuilds the spine
assembly manifest from its declared `source_channels` parameter and the live
provenance columns. The row-count arm of that manifest is self-consistent by
construction and is not evidence; what stays independent is the declared-channel
roster (an undeclared live channel is refused), the clone-index domain, and the
cross-grain check that each person's support channel matches every group row
that person references. Drift *between* stages is caught by the executor's own
content addressing, which is what the manifest exists to do upstream.

## Receipt

The stage publishes counts, mass totals, declared labels and 64-hex digests
only: per-entity base/native/detail/expanded row counts, a lineage digest and a
native-source-id digest per entity, per-channel and per-role provenance counts
before and after, household mass in total and per source channel, the clone
attachment authority receipt, and explicit `donor_inputs` flags recording that
no donor was read, no donor weight entered a host weight, no draw or
tax/benefit value was written and no prior-wage stage ran.

The executor authors the rest of the node receipt itself. That includes
`receipt["expand"]`, the per-entity `[[new id, source id], …]` lineage it needs
to restore a cached population without re-running the kernel. That row-level
record is executor machinery, not this stage's published evidence, and it
cannot be suppressed without giving up warm restore.

## Usage

```python
expand, claim = us_combined_survey_clone_nodes(
    columns,                      # the base version's column inventory
    base=combined_population,     # the combined native host version
    source_channels=("acs", "asec"),
)
register_us_combined_survey_clone_kernels(registry)
```

The next stage consumes the `EXPAND` node's population. Recipient feature
construction, donor admission, fitting, drawing and placement are separate,
separately reviewed scopes.

## Verification

`packages/microcosm-build/tests/test_us_graph_combined_clone.py` executes the
stage through the real executor over an invented mixed ASEC/ACS host assembled
by the real `assemble_spines`: two arms whose raw record ids deliberately
collide, five households, ten people, seven tax units, two tax units inside one
household, one zero-weight household, unequal weights, non-contiguous ids and a
person order that is not the id order. No microdata is read and no donor
exists.
