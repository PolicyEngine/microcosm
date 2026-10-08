import assert from 'node:assert/strict';
import { parseGraphDocument } from '@axiom-foundation/orrery';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const document = parseGraphDocument(JSON.parse(Buffer.concat(chunks).toString('utf8')));
assert.ok(document.nodes.some(node => node.kind === 'group'));
assert.ok(document.nodes.some(node => node.parentId));
assert.ok(document.nodes.some(node => node.sources?.length));
assert.ok(document.nodes.some(node => node.statuses?.length === 3));
assert.ok(document.activities.length);
assert.ok(document.artifacts.length);
assert.ok(!document.receipts?.length);
process.stdout.write('Orrery accepted grouping and recorded execution.\n');
