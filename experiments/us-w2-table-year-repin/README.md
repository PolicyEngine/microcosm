# W-2 item tables: re-pin the US feed to the tax year IRS published

10 October 2026. The pinned US Chronicle feed stamped five cells of IRS SOI
Table 4.B for Tax Year 2020 as ty2023. `docs/us-chronicle-feed-repin.md`
("The W-2 item tables carry Tax Year 2020") has the account; this folder holds
the scripts and their outputs.

| File | What it does |
|---|---|
| `relabel_scope.py` | Rewrites `us/chronicle_feed_scope.json`: the three ty2023 W-2 pairs out, the TY2020 401(k) and Roth record sets in, the new Chronicle commit. Idempotent. |
| `splice_feed.py` | Predicts the new feed from the old one and one run of the W-2 package (`docs/us-fact-to-target.md`, step 1). |
| `apply_pin.py` | Writes the rebuilt feed's digests into `us/chronicle_feed.json` and the parity generator's feed name, from the builder's receipt. |
| `compare_feeds.py` | Row-by-row difference between two feeds. Output: `feed_compare.json`. |
| `measure_surface.py` | Compiles a feed as the release does and lists every target. |
| `compare_surfaces.py` | Target-by-target difference between two `measure_surface.py` reports. Output: `compare.json`. |
| `receipt.json` | The receipt `tools/build_us_chronicle_feed.py` wrote for the pinned feed. |

## Order of work

```bash
python experiments/us-w2-table-year-repin/relabel_scope.py packages/microcosm-build/src/microcosm/build/us/chronicle_feed_scope.json f98acf4edcc9488343446fda46db77914de8611f
```

```bash
uv run python tools/build_us_chronicle_feed.py --chronicle-root <chronicle checkout at microcosm-us-feed-w2-ty2020-v1> --out <feed dir> --replace --skip-artifact
```

```bash
python experiments/us-w2-table-year-repin/apply_pin.py . <feed dir> f98acf4edcc9488343446fda46db77914de8611f consumer_facts_us_f98acf4.jsonl
```

Then copy `<feed dir>/consumer_facts.jsonl` to
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_f98acf4.jsonl` and run
`uv run python tools/build_us_target_parity_manifest.py`.

## Results

`feed_compare.json`: of the old feed's 39,158 rows, 39,153 are byte-identical
in the new feed; the five ty2023 W-2 ids are gone; two ids are new (the
401(k) and Roth amounts at ty2020), each with the value and source cells of
its ty2023 twin. The new feed has 39,155 rows.

`compare.json`: "before" is main `7639ef8b0` compiling the old feed, "after"
is this change compiling the new one. Both compile 32,842 targets, 5,694 of
them on `national_state`. One target differs: the W-2 Box 7 tips amount is
`irs_soi.ty2023.…amount` at $28,280,884,269.41 before and
`irs_soi.ty2020.…amount` at $34,287,530,778.93 after. The other 32,841 match
on value and on a digest of all their metadata.

"Before" has to run on main's code, because this change refuses the old
feed. It ran from `git archive origin/main packages/microcosm-build/src tools`
with that tree first on `PYTHONPATH`; the report's `compiler` field recorded
which `fiscal_targets.py` did the compile.

The splice and the builder agree: `splice_feed.py` on the old feed and one
`chronicle build-bundle --year 2020 --source packages/irs_soi/w2_statistics_2020`
run at `f98acf4` gives the bytes `tools/build_us_chronicle_feed.py` writes
from the new scope (`facts_sha256` `a81cbcc5…23df04c`).
