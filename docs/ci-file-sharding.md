# CI file sharding

The fast `rest`, engine `us-am` and installed-wheel inventory each run across
four independent GitHub Actions matrix jobs per Python version. The initial
partition uses sorted file paths round-robin. It balances file counts, not
measured runtime; an expensive single module can still dominate a job.

Files are the smallest scheduling unit. All tests in a module, its module-scoped
fixtures, cold/required pairs and mutation controls stay in the same pytest
invocation. `us-am:build` and `us-am:other-shards` remain separate processes
inside each of the four jobs, preserving import isolation. The US `us-qs` and
UK process divisions also remain unchanged.

The environment tiers keep their original scope:

- Fast covers the existing inventory except the explicit `ENGINE_ONLY` list.
- Engine groups partition the full inventory, with the existing country change
  filters and US/UK extras. Both Python versions stay unchanged.
- Wheels partition the complete inventory, including `ENGINE_ONLY` files. Their
  engine markers/import guards still skip when appropriate. Each matrix job
  builds every real wheel, installs into a clean constrained venv, checks import
  and spec identity, installs the same source extras, performs the HDF smoke and
  confirms engine absence before its portion of the full suite.

No tests are renamed, newly marked, skipped or replaced by sharding. Only the
partition regression file is added. The independent seed-diagnostic job and
the required `ci-ok` aggregation are unchanged. More jobs repeat setup and
wheel-building work; shorter wall time is not guaranteed by this configuration.

## Selection and verification

`tools/ci_test_groups.py` owns the file partition. For example:

```sh
python3 tools/ci_test_groups.py --list rest --shard 1/4
python3 tools/ci_test_groups.py --list us-am:build --shard 2/4
python3 tools/ci_test_groups.py --list wheels --shard 4/4
```

`INDEX/COUNT` is 1-based, with `1 <= INDEX <= COUNT <= 64`. Selection sorts
the files and takes `files[INDEX-1::COUNT]` after the group/process filter.
Duplicate inventory entries, invalid ranges and an inventory too small to fill
every requested shard fail. Omitting the option preserves the original full
group selection. `wheels` is deliberately separate from the fast classifier.

The workflow uses a checked command substitution and rejects empty output
before turning it into a bash array. It does not hide selector failures inside
process substitution, which can otherwise produce an empty pytest invocation
and inadvertently run the entire suite.

```sh
python3 tools/ci_test_groups.py --verify
python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py
```

These commands need only the standard library and Git; they do not import
Microcosm, collect data tests, build wheels or run a microsimulation. The first
proves the nonempty, disjoint unions across groups, processes and configured
shards and rejects nested tests. The second exercises invalid/empty/duplicate
selection and verifies the actual workflow matrix covers each tier exactly
once while retaining the Python versions, process isolation and wheel boundary.

Adding a file can shift subsequent round-robin assignments. The coverage proof
must still pass on the current checkout. Future tuning should use separately
reviewed module-level timing evidence; file counts are not timing evidence and
must not justify weaker checks or longer timeouts.
