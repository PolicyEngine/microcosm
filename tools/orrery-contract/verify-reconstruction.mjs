import assert from 'node:assert/strict';
import { parseGraphDocument } from '@axiom-foundation/orrery';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const document = parseGraphDocument(JSON.parse(Buffer.concat(chunks).toString('utf8')));
const marker = document.metadata.microcosm.extensions['microcosm.reconstruction'];
assert.equal(marker.protocol, 'microcosm.graph.reconstruction.v1');
assert.equal(marker.reconstructed, true);
assert.equal(marker.basis, 'declaration_only');
assert.match(marker.source_revision.git_commit, /^[0-9a-f]{40}([0-9a-f]{24})?$/);
assert.equal(marker.inputs.filter(input => input.role === 'declaration').length, 1);
assert.equal(
  marker.inputs.find(input => input.role === 'declaration').sha256,
  document.metadata.microcosm.graph_sha256,
);
assert.ok(document.nodes.some(node => node.kind === 'operation'));
// A declaration-only reconstruction never presents an operation as executed.
assert.ok(!document.nodes.some(node => node.statuses?.length));
assert.ok(!document.activities?.length);
assert.ok(!document.artifacts?.length);
assert.ok(!('execution' in document.metadata));
process.stdout.write('Orrery accepted the reconstructed declaration.\n');
