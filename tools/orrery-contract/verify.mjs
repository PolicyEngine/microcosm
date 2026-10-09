import assert from 'node:assert/strict';
import { parseGraphDocument } from '@axiom-foundation/orrery';

const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const document = parseGraphDocument(JSON.parse(Buffer.concat(chunks).toString('utf8')));

assert.equal(document.schemaVersion, 'graph-explorer/v1');
assert.equal(document.metadata.adapter, 'microcosm.graph.orrery.v1');
assert.ok(document.nodes.some(node => node.kind === 'operation'));
assert.ok(document.nodes.some(node => node.kind === 'source'));
assert.ok(document.nodes.some(node => node.kind === 'field'));
assert.deepEqual(
  [...new Set(document.edges.map(edge => edge.kind))].sort(),
  [
    'artifact_input',
    'compiled_predecessor',
    'declared_read',
    'declared_source',
    'provided_field',
    'structural_input',
  ],
);
assert.deepEqual(
  [...new Set(document.edges.map(edge => edge.category))].sort(),
  ['dependency', 'provenance'],
);
assert.deepEqual(document.metadata.microcosm.extensions, {
  float: { float_literal: '1e+100' },
  large: { integer_literal: '1208925819614629174706176' },
  negative: { integer_literal: '-1208925819614629174706176' },
});

const materializedRead = document.edges.find(
  edge => edge.kind === 'declared_read'
    && edge.data?.read_kind === 'materialized_expand_output',
);
assert.ok(materializedRead);
const materializedField = document.nodes.find(node => node.id === materializedRead.source);
assert.equal(materializedField?.kind, 'field');
assert.equal(materializedField?.data?.provider, 'expanded');
assert.equal(materializedField?.data?.declared_in, 'expanded.claim');
assert.equal(materializedField?.data?.visible_in_schema, false);
process.stdout.write('Orrery accepted the Microcosm graph document.\n');
