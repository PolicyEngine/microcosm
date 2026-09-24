# US base build: pinned CPS ASEC inputs

The base stage of `tools/build_us_puf_support_base.py` reads three processed
CPS ASEC files, one per income year 2022, 2023 and 2024, through
`--asec-h5 YEAR=PATH`. Until 18 September 2026 those files existed in one
untracked directory on one build machine, and the build recorded their
digests without comparing them to anything.

## Where the files live

They are mirrored, unchanged, to the public Hugging Face dataset repository
[`policyengine/microcosm-us-sources`](https://huggingface.co/datasets/policyengine/microcosm-us-sources)
(CC BY 4.0; they derive from public Census Bureau files). Microcosm addresses
them by upload revision, never by branch, so a later upload cannot move what
a build reads.

`microcosm.build.us_runtime.asec_sources` owns the coordinates:

| Income year | File | SHA-256 | Bytes |
|---|---|---|---|
| 2022 | `census_cps_2022.h5` | `7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e` | 301,129,278 |
| 2023 | `census_cps_2023.h5` | `cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88` | 299,036,610 |
| 2024 | `census_cps_2024.h5` | `ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d` | 323,994,739 |

Revision `78acc83ea8b099a97cb0d658bbed91ea75aae8b0`. The same digests are
recorded independently as hermetic-input evidence in
`ecps_parity_known_gaps.json`, and `test_us_asec_sources.py` asserts the two
rosters agree, so there is one source of truth.

## Fetching

`fetch_asec_source(year)` returns a verified local path. It checks the archived
US data repository checkout's storage directory and
`~/.cache/microcosm/asec/` first, to spare a 300 MB transfer on machines that
already hold the file, then downloads the pinned revision through
`huggingface_hub`. Every candidate, the download included, must match the
pinned byte length and SHA-256 before it is returned; a download that does not
raises `ValueError`. Passing `cache_dir` skips the convenience copies and
downloads into that `huggingface_hub` cache.

`tools/fetch_us_asec_sources.py` resolves all three and prints the builder's
arguments, one year per line:

```bash
uv run python tools/fetch_us_asec_sources.py
```

## Verifying in the build

`--asec-h5-sha256 YEAR=SHA256`, passed once per year, makes the builder refuse
a `--asec-h5` input whose bytes differ from the declared digest. The check runs
in `main()` before any stage, so a wrong input refuses in seconds rather than
surfacing as drift hours later. Every pin is parsed and checked before any
file is read: a pin without `--asec-h5`, a value that is not `YEAR=SHA256`, a
year that is not an integer, a digest that is not 64 hexadecimal characters, a
year named twice, a year no `--asec-h5` mapping provides, and, for a year with
a canonical pin, a declared digest that differs from it (the flag can narrow
nothing and re-pin nothing) all refuse with a message naming the value. Once
any year is pinned, every `--asec-h5` year must be pinned, so a build is either
fully pinned or not pinned at all. Then each pinned file is hashed in year
order; a missing file and a digest mismatch refuse the same way. The pins are
locked into the checkpoint `run_config` (`asec_h5_sha256`, a `{year: digest}`
mapping with the digest lower-cased) so a resume with different pins refuses
and an equivalently spelled pin resumes, and the raw values are forwarded to
every staged child process.

The flag is opt-in. Fixture-driven tests build from synthetic
`census_cps_*.h5` files and are not held to the production digests. A
certified from-scratch build passes it for all three years; nothing enforces
that yet, and the US release build rule is where it belongs.
