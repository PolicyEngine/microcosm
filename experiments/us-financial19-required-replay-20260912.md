# Native financial graph required replay — 12 September 2026

The frozen nineteen-node financial graph passed a separate required-cache
replay on real ACS/ASEC inputs. This accepts replay of the earlier financial
pilot, not PUF enrichment, a calibration, a current-source population or release.

## Exact scope

- Source: `2ca11c85a0f710b8fa495e363243b940f7eabae2`.
- One collected test passed; outer guard result `0`.
- Request: source fraction `1/1000`, survey/geography seed `20260908`, two
  trees, demographic conditioning enabled, `resume="require"`.
- All nine prefix nodes and all nineteen financial-graph nodes were cache hits.
- The graph, preparation, projection and recipient-matrix exports matched the
  cold run byte for byte. Portable manifest content matched after removing only
  start/end times and per-node timing/cache-hit fields. The complete artifact
  report also matched, excluding its cache-hit and manifest-file identity fields.
- Source, resources, native inputs, owned files, baseline exports and thread
  controls remained unchanged. There were no unexpected access refusals.
- Elapsed: `3981.09802937496` seconds; CPU: `3944.4073700000004` seconds;
  peak RSS: `10988191744` bytes. Declared limits were 7,200 CPU seconds,
  10,800 elapsed seconds and 32 GiB peak RSS, with one numerical worker.

The source population remains 1,584 households / 3,464 people, expanded to
3,168 households / 6,928 people before block assignment. The final stage carries
seven financial fields. The same frozen source produced the 11 September cold
pilot; no broader coverage follows from replay.

## Evidence identities

| Evidence | SHA256 |
| --- | --- |
| Closed replay guard receipt | `71d6981518acad0ad0fd199f906c01ccfa867796aaf053cf6fa5f5a3df1193ae` |
| Final replay manifest | `c8a1b7715bae740c183aa592fca718849e35edf6722328c1de3d83f0cacafa77` |
| Cold baseline manifest | `0d745678802d8dae16750f57ebe779644237e1b2416f1cb78edf4c59d2163f0e` |
| Reviewed guard | `5a2bcd05665644a86ff2f3fc35a2e7e751d759412380dce07193f9b0c6f3c670` |
| Manifest content key, both runs | `134ab4ea39b434066e48258be345999425a6d23f3b5016680f457fd357be9edf` |

Retained local evidence is in
`PolicyEngine/_recovered/pilot-runs/native19-required-20260912/run`:
`us-native-postclone-financial19-required-20260912.json`,
`native-financial-manifest.json`, and the reviewed harness and helper.
The coordinator's metadata postcheck is
`PolicyEngine/_recovered/scratch-backup/893/codex-takeover-20260912/native-required-replay-postcheck.json`.
It rechecked receipt consistency, exact owned-file hashes, reported bounds,
node hits and export identities without reading native data or generated Frame
payloads again. The actual payload comparison and final source/output checks ran
inside the reviewed native harness before its successful guard closure.

The guard's expected optional dependency probes remain recorded separately from
unexpected refusals. The run enabled no engine, network, child processes or
automatic retry. Original cold and failed-attempt evidence remains retained.

## Next acceptance

The maintained integration includes later runtime repairs and the restored
PUF55 host. Its native candidate needs a fresh cold execution, required replay,
complete input coverage and full Frame export/readback on that identified source.
Calibration and complete-population acceptance follow separately. This replay
does not authorize reuse of the serialized receipt as a live population issuer.
