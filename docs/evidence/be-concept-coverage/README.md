# Belgian pilot input inventory snapshot

Generated on 2026-08-29 by the shared diagnostic tool, using the real Axiom dense
compiler. [pilot-inputs.json](pilot-inputs.json) contains seven Person inputs,
their runtime-provided canonical addresses, and six unknown metadata fields per
input. It supplies **no dataset and no evidence bindings**. Household compilation
reports `no_derived_program` and a null count, not successful zero enumeration.
It is not a complete Belgian-law inventory or a certified population schema.

The two identical runs produced content digest
`8b6c379afbf21e590c259722207ea07e41043be6ef335b6f77fed9a49b553086`.
The manifest fingerprints the actual adapter/builder files, installed engine
wrapper/native package, RuleSpec entry module, and canonical root YAML/toolchain.
The producer-code digests identify the files used for this dated snapshot, not
an attestation of any published release.

## Source and build pins

- Microcosm base: `ba73e2f43c6f1dcb3533faeab26ecc18260bd712`.
- Axiom source: `bb4b5684870547756078a62f1866a77c5b56f7f3`.
- RuleSpec-BE source: `b105e2b3a3086ddd2de447d58a9b951346870dd1`.
- Entry module: `be/statutes/income_tax/individual/pilot_worker_oracle_pipeline.yaml`.
- Native wheel: `axiom_rules_engine_dense-0.1.0-cp314-cp314-macosx_11_0_arm64.whl`,
  SHA-256 `7928e395942ae071054197891a656f4744864e57c07554a9d0bf7d6635fe0f90`.
- Effective native Cargo lock SHA-256:
  `e40add4c7ddeb39fa0005f379abc438bf9d2d927ac4570e8f6bf45d671a9e1b2`.

The pinned source's extension lock was stale: its local path dependency named
engine version 0.1.0 while the source crate declared 0.2.2. The initial `--locked`
build refused. `maturin build --release --offline` changed that one local-crate
version entry; no external dependency version changed. The wrapper/native
distribution labels remain 0.1.0, so they must not substitute for the file
fingerprints. The manifest leaves the unexposed runtime core version null.

Build/runtime outputs live outside tracked source. This snapshot is unsigned
diagnostic evidence, not an attested binary build or signed-corpus release.

## Reproduction

Install the pinned wrapper and built native wheel into a Python 3.14 environment
containing these Microcosm sources. The exact command surface is:

```bash
uv run --no-sync python tools/inventory_axiom_concepts.py \
  --module /absolute/path/rulespec-be/be/statutes/income_tax/individual/pilot_worker_oracle_pipeline.yaml \
  --rulespec-root /absolute/path/rulespec-be \
  --group-entity household
```

The Work Bonus input's canonical owner is its imported `work_bonus` module, not
the pilot entry module. The other six addresses belong to the pilot module.
Those addresses came from the runtime catalog, not string construction. Runtime
inputs include configuration rates and supplied derived quantities as well as
remuneration; this diagnostic does not require every input to be observed data.

## Separate target-inventory observation

The local pilot artifact `microcosm_be_v051_chronicle_targets.json`, SHA-256
`250051e7c3e50a4aee5181857c72d78f4ec38c365ff7dd1fb6c00ae6def7c227`,
declares 956 targets and 91 validations. The session's scoped inventory audit
found no wealth-stock targets. A reproducible name-screen returned no target
names containing `wealth`, `net_worth`, `asset`, `mortgage`, `debt`, or `hfcs`.
This is an audit of that specific artifact, not a classification engine or a
claim that Belgian publishers lack wealth statistics. The input diagnostic does
not ingest or republish that target file, and no source-to-input equivalence is
asserted here. Income and saving flows do not establish wealth-stock coverage.
