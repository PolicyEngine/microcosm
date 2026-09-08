import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { validateGraphDocument } from '/Users/maxghenis/TheAxiomFoundation/graph-explorer/src/core/validate.ts';

const root = '/private/tmp/us-shared-explorer-adapter-20260908';
const bytes = await readFile(`${root}/actual-four-node.graph-explorer.json`);
const document = JSON.parse(bytes.toString('utf8'));
const result = validateGraphDocument(document);
const summary = {
  valid: result.valid, issues: result.issues,
  documentSha256: createHash('sha256').update(bytes).digest('hex'),
  nodes: document.nodes.length, edges: document.edges.length,
};
await writeFile(`${root}/shared-validation.json`, JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary));
if (!result.valid) process.exitCode = 1;
