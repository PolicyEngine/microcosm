# Axiom input-surface snapshots

`nz.json` and `be.json` record, for each RuleSpec module in rulespec-nz and
rulespec-be, the input slots the Axiom engine accepts for each engine entity.
They are generated from the real engine at pinned commits by
`tools/refresh_axiom_input_surface.py`. Engine-free tests and `tools/refresh_concept_coverage.py` use them in place of
a live engine: CI installs no Axiom engine. They are test fixtures, not
runtime data, so they do not ship in the microcosm-frame wheel. Never hand-edit these files.
Regenerate them and review the diff.

## Contents

Each file has format `microcosm.axiom_input_surface.v1`. Keys are sorted,
the indent is 1, and the output is byte-identical across runs.

- `rulespec.repository` and `rulespec.commit` give the rulespec repository and
  the full SHA that was compiled. The tree was checked byte for byte against
  that commit before compiling.
- `engine.repository`, `engine.commit` and `engine.version` identify the engine
  build. `engine.version` is the CLI's `--version` output.
- `engine.surface` names what produced `inputs`:
  - `dense_root_inputs`:
    `CompiledDenseProgram.from_file(module, rulespec_roots=[root], entity=E).root_inputs`.
    This is what `AxiomEngine.variables()` reads. A walk of the CLI artifact
    was run as a differential check.
  - `cli_artifact_walk`: a walk of the CLI compile artifact only.
- `modules[]` has one entry per non-test `*.yaml` under the country trees
  (`nz`; `be`, `be-bru`, `be-dg`, `be-vlg`, `be-wal`). Each entry has:
  - `path`: relative to the rulespec root.
  - `sha256`: of the module file bytes.
  - `status`: `compiled`, `compile_failed` or `dense_unsupported`.
  - `inputs`: engine entity to its sorted root inputs. An entity is listed only
    if it has a dense program. `Scalar` is the engine's pseudo-entity for
    entity-less formula rules.
  - `canonical_inputs`: slot to the artifact's `canonical_request_name`.
    Canonical names are module-scoped: the same slot can have a different
    canonical name in another module.
  - For `dense_unsupported` modules only:
    - `dense_unsupported_entities`: the entities the dense compiler cannot
      build.
    - `cli_walk_inputs`: the CLI-walk inputs of those entities. The adapter
      cannot bind them.
    - `error`: why dense compilation fails. It is deterministic: the
      lexicographically first blocker, plus a count of the rest.

## Regenerate

Run from the repository root. `AXIOM` is the directory holding the
`axiom-rules-engine`, `rulespec-nz` and `rulespec-be` checkouts; run
`git fetch origin` in each first. Everything is built from `git archive`
output in a scratch directory, never inside a checkout.

```bash
W=<empty scratch dir>
AXIOM=<dir with axiom-rules-engine, rulespec-nz, rulespec-be checkouts>
ENGINE=04315d94d04efd61e27665ab3aa300d64fdf3a66
NZ=6fe181fc4c65763150fe97501ef3443774e64869
BE=b105e2b3a3086ddd2de447d58a9b951346870dd1
mkdir -p $W/engine $W/rulespec-nz $W/rulespec-be
git -C $AXIOM/axiom-rules-engine archive $ENGINE | tar -x -C $W/engine
git -C $AXIOM/rulespec-nz archive $NZ | tar -x -C $W/rulespec-nz
git -C $AXIOM/rulespec-be archive $BE | tar -x -C $W/rulespec-be
(cd $W/engine && cargo build --release --locked)
uv venv --python <python3.14> $W/dense-venv  # any GIL-enabled CPython 3.14
VIRTUAL_ENV=$W/dense-venv uv pip install maturin
(cd $W/engine && PYO3_PYTHON=$W/dense-venv/bin/python \
  $W/dense-venv/bin/maturin build --release --manifest-path python-ext/Cargo.toml \
  -i $W/dense-venv/bin/python --out $W/wheels)
VIRTUAL_ENV=$W/dense-venv uv pip install $W/wheels/*.whl $W/engine/python
for CC in nz be; do
  SHA=$([ $CC = nz ] && echo $NZ || echo $BE)
  uv run --no-sync python tools/refresh_axiom_input_surface.py \
    --country $CC --rulespec-root $W/rulespec-$CC --rulespec-commit $SHA \
    --rulespec-git-dir $AXIOM/rulespec-$CC \
    --engine-bin $W/engine/target/release/axiom-rules-engine \
    --engine-commit $ENGINE --dense-python $W/dense-venv/bin/python
done
```

Add `--check` to compare a fresh run with the checked-in file without
writing. It exits 1 and summarizes the differences when they differ.

`maturin build` runs without `--locked` for one reason. At the pinned engine
commit, `python-ext/Cargo.lock` still records the path crate
`axiom-rules-engine` as 0.1.0, and the crate is 0.2.2. The extension must be
built for the interpreter that loads it: a wheel built for free-threaded 3.14t
does not import on GIL-enabled 3.14.
