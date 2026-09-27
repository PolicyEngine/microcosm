# Joint empirical graph operations

The reusable adapter fits a weighted paired empirical law and draws from it with
the maintained keyed-uniform protocol. It uses `joint_empirical.py`; it does not
fit QRF or independently sample two marginals. The source host selects and
authenticates donors and recipients, supplies an explicit support policy, and
independently verifies draws before attaching them to a survey population.

This source implementation depends on the numerical operator introduced in
`aaa8b071ab14a97e514deb05d367f74a964abb29`. Its tests are authored but unrun.

## Private support branches

The 2026-09-13 root decision clarifies the initial child implementation plan:
independent private model-support CREATE branches are allowed so their exact
column schemas appear in graph Slices. They do not change the receiving survey
population, original spine, clones or geography. There is no new EXPAND operation.
The US source owner creates those branches from its retained qualified source.
The reusable adapter owns only the fit and draw operations.

Side-table entity IDs must be exact unique int64 values. They identify rows when
results are joined back and are excluded from the declared random coordinates.
Original source coordinates occupy separately declared int/string columns. No
float or Boolean source coordinate is accepted. The host must validate both the
original coordinate and the original-person join; a matching side-table ID alone
does not confer source authority.

## Fit contract

`joint_empirical_fit_node` declares one all-row Slice with the two ordered target
columns, original donor and household key columns, and explicit weight column.
Targets and weights are nonnullable float64; original keys retain their integer
or string types. Donor keys must be unique, while household keys may repeat.
All numeric validation and support rules execute in the real numerical model.
The input weight kind is recorded as `explicit`; the host separately documents
and authenticates any original household design-weight provenance.

The node consumes a typed `source_projection` artifact whose payload hash is an
explicit parameter. This is the full donor projection, distinct from the selected
recipient projection. The complete support document is a canonical JSON string
plus its SHA256 in normative parameters. No production support default exists.

The node emits the numerical operator's exact private `model` bytes and a typed
`model_metadata` artifact with model/support/donor-source hashes, target names,
explicit weight kind and aggregate diagnostics. Model bytes contain paired donor
values and exact tagged private coordinates. Metadata contains no donor IDs.
Changing selected recipient sampling does not enter the numerical model bytes.

## Draw contract

`joint_empirical_draw_node` declares the original-recipient coordinate Slice. It
can also declare an exact nonnullable Boolean `eligibility_column`. The support
Frame then contains all real original recipients; the kernel selects eligible
coordinates internally, with candidate/draw counts in its receipt. An all-false
column produces a legitimate zero-row draw while retaining the nonempty support
Frame. This uses no sentinel person and does not relax Frame or Weights: a
row-masked empty Slice would itself require forbidden empty projected weights.
An empty direct DataFrame test is only a numerical context test, not this graph
path. Invalid or duplicate unselected coordinates are still refused.

The node
consumes model and metadata from the same producer and the separate typed
recipient `source_projection`. It checks platform scope, artifact store identity,
the numerical model's complete plain-data encoding, support agreement, donor
source hash and recipient projection hash.

Parameters include the complete stream `(algorithm, experiment_id, replicate,
base_seed)`, a coordinate suffix, the complete canonical transport document and
hash, and explicit source/support/scenario revisions. The US child candidate
uses suffix `('child-property-v1',)`. The adapter appends `pattern` and `donor`
separately and calls the real `keyed_uniform`. It declares `SeedSource.KEYED` and
does not consume the node-key-seeded sequential generator.

Only the original source coordinates and explicit suffix enter the stream.
Side-table IDs, clone numbers, row position, batch boundaries, model node keys,
recipient sample size and scenario hashes do not. The scenario hash is a host
revision reference, not authority over its semantics; the actual transport and
stream parameters are independently normative. Scenario comparisons share
uniforms while remaining distinct content-addressed graph operations.

The private `draws` artifact records exact tagged original coordinates, decimal
int64 side IDs, paired float64 values, patterns, positive donor references and
all model/support/source/scenario/transport bindings. Rows have canonical
coordinate ordering. Pattern zero has no sampled donor identity; it remains a
model zero rather than an observed nonreceipt. No survey columns are written.

`read_joint_empirical_draw` checks exact supplied side IDs and original keys plus
the expected hashes and parameter documents. It returns values in the supplied
order as bytes-backed readonly arrays. This validates the serialization and join
contract, not the draw law: a host must reconstruct the actual model and keyed
draws from retained qualified source before accepting canonical attachment.

## Verification and limits

The invented tests cover actual numerical fit/draw calls, unequal-weight paired
probabilities, exact large IDs, integer/string distinctions, row permutations,
partitions, recipient additions, side-ID changes, common scenario uniforms,
typed artifact/producer/hash changes, malformed private joins, explicit zero
stress and empty recipient batches. A real four-node cold/required replay checks
both model-support Frames and scenario-specific draw keys. No test grants survey
authority, and no native fit or production support policy is claimed.

The numerical model is capped at 64 MiB and 1,048,576 donors. Private draw JSON is
also capped at 64 MiB and 1,048,576 recipients; the byte check occurs after Python
object construction. These are ceilings, not measured national capacity. Before
large use, measure bytes per row, working-memory peaks and repeated validation
cost on small samples. A separately reviewed compact transport may be needed.
Implementation identity includes this adapter, the numerical model, weight
resolver, keyed-randomness module and its canonical serializer implementation,
with NumPy/Pandas platform-bitwise scope. Serialized metadata and draw-header
bindings compare canonical bytes, preserving integer/Boolean/float distinctions.

The corrective source tests add valid changed-model/stale-metadata refusal,
noninteger encoded stream/count refusal, canonical source identity and actual
compiler-key changes for source/support/transport/stream revisions. The actual
four-node runtime test now covers all, some and zero eligible recipients with
cold/required replay and refusal of corrupted invented cache payloads. These
tests are authored and UNRUN pending a separately approved bounded runtime.
