import assert from 'node:assert/strict';
import { parseGraphDocument } from '@axiom-foundation/orrery';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const document = parseGraphDocument(JSON.parse(Buffer.concat(chunks).toString('utf8')));

assert.equal(document.schemaVersion, 'graph-explorer/v1');
assert.equal(document.metadata.adapter, 'microcosm.graph.orrery.v1');
assert.ok(document.nodes.some(node => node.kind === 'operation'));
assert.ok(document.nodes.some(node => node.kind === 'field'));
assert.ok(document.edges.some(edge => edge.kind === 'declared_read'));
process.stdout.write('Orrery accepted the Microcosm graph document.\n');
