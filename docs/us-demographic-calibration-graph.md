# National demographic calibration in the graph

`demographic_calibration_node` appends a development calibration to an existing
population version with household importance weights. It uses the real Adam
solver and writes its full version-6 diagnostic payload as a typed graph
artifact. The existing `calibrate.adam@1` contract remains unchanged.

The caller supplies a `TargetRegistry` for explicitly declared 2024 national
population counts from S0101 or B01001, with each person's contribution already
aggregated to their household. Source acquisition, population-universe alignment
and the decision to use a reference for calibration happen before this seam.
A reference capture is not automatically an activated registry. This module
checks declared semantics; it does not independently authenticate citations or
prove that a source-to-measure mapping is scientifically appropriate.

Each target names its publisher count cell, household count column, period,
source, family and metadata: `table`, `reference_sha256`, `geography`, `universe`,
`role`, and `evidence_scope`. Geography is the direct national row `0100000US`,
universe is `population`, and role is `calibration`. `evidence_scope` separates
invented fixtures from source-documented inputs. A registry cannot mix the two.
The complete canonical registry, including these declarations, enters the node
key. Mutating an original registry after constructing a node cannot change the
node's frozen declaration. B19001 and B25003 are reserved and are refused.

The node reads only its declared household count columns. Missing, negative and
fractional counts refuse execution; Boolean indicators are valid zero/one counts.
Incoming importance weights must be strictly positive, and skipped targets refuse
the result. The same household weight applies to all
people in the household; this module does not introduce independent person or
tax-unit weight vectors. Count preparation must preserve every included person's
membership and make any excluded or unknown universe explicit upstream.

Two weight caps have different anchors:

- `max_initial_weight_ratio` bounds the solver relative to incoming importance
  weights, which may already reflect stacked-spine harmonization.
- `max_weight_ratio` bounds the graph result relative to original design weights,
  retained by the executor from the source population. A missing anchor or
  violated cap refuses the result. The kernel does not secretly access that
  earlier population or relabel one anchor as the other.

The returned weights have kind `calibrated`; graph execution preserves all
population columns and memberships. The diagnostic artifact includes per-target
before/after estimates, source metadata, loss history, weight concentration,
an empty skipped-target list and diagnostic warnings. Its `build` metadata
names both cap anchors and values, so an artifact reader does not have to infer
them from the node receipt. The artifact type follows the diagnostic payload's
schema constant. Standard errors travel with the target
rows but are **not consumed by the current loss**. Calibration fit is distinct
from a holdout test. Neither this node nor a successful replay certifies national
or congressional-district analysis.

The first fixture has four fictional households and six people. Source design
weights become different importance weights before calibration. Additional
fixtures verify rejection when the solver's allowed result exceeds the original
design cap, and when a source has no original design anchor. The real graph result matches
the direct solver and diagnostic payload; a fresh Python process replays all
three nodes from the content store with every kernel forbidden from running.
No Census values, national registry activation or genuine population calibration
are established by that fixture.

Descriptions and citations remain descriptive: they do not change a node's
normative declaration or prevent execution.

This slice does not rewrite a US frame-context artifact from an earlier
population version. Downstream consumers should use the executor-owned calibrated
population and the diagnostic edge. A consumer requiring the older US context
contract needs an explicit successor context with its new weight kind and mass
record before it can be attached. Pruning and a national/CD release remain later
composed graph stages.
