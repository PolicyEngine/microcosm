# US head-to-head slice times, measured locally

What this measures: how long the pinned scorer's slice work takes, chunk by
chunk, for the d844 incumbent and candidate, and how its worker pool behaves.
It is the d844 run's sizing input and the source of `chunk-weights.json`,
which the early-stop estimate (`tools/us_head_to_head_projection.py
--chunk-weights`) uses for chunks it has not measured yet. Three parts: slice
times in one process (below), a three-worker pool run of the scorer itself,
and simulated 16-worker runs, which are not measurements.

## Slice times in one process

- Scorer: `tools/score_us_release_head_to_head.py` at 2cfe3043d, the plan's
  pin, in a detached checkout.
- Interpreter: CPython 3.14.7, the standard build, with the pin's locked
  dependencies (policyengine-us 2.2.1). BLAS and OpenMP pools at one thread.
- Inputs: incumbent H5 sha256 6496cc43… (29 slices of 2,000 households);
  candidate H5 sha256 2d9c8586… (the d1386 build, 177 slices); ledger
  `consumer_facts_us_c5e5bf8` sha256 b8543739…; 33,147 specs in 5 chunks of
  8,192 (chunk 5: 379 specs, including the 11 JCT reforms).
- Host: an 18-core Apple M5 Max shared with other work (1-minute load 30 to
  120 during the run). 2026-11-16, 18:08 to 18:36 UTC.
- Method (`scripts/time_slices.py`): one process loads the yardstick and both
  artifacts once, repairs each frame's population mass as the scorer does,
  and calls the scorer's `_score_household_slice` with the arguments its
  workers pass. One warm-up slice per artifact, then, for each chunk, two
  slices per artifact in alternation (incumbent slices 8 and 22, candidate
  slices 45 and 133), so load drift hits both alike. The report is rewritten
  after every slice.

### Results

Seconds per slice, wall clock; CPU time was 94% to 99% of it (92% and 95%
in the two warm-ups).

| Chunk | Candidate | Incumbent |
|---|---|---|
| 1 | 35.9, 41.9 | 49.9, 33.3 |
| 2 | 42.4, 40.1 | 49.2, 42.9 |
| 3 | 37.2, 39.0 | 43.8, 40.4 |
| 4 | 34.4, 34.9 | 47.5, 41.2 |
| 5 | 202.1, 197.3 | 232.2, 225.8 |

Warm-up slices (chunk 1): candidate 41.2, incumbent 53.6. Import took 38.1
s, the yardstick compile 22.5 s, and loading the candidate 11.3 s. The one
process holding both frames read 7.5 to 9.3 GiB current RSS (psutil) after
each slice; the highest psutil sample, every half second, was 10.4 GiB. Its
`ru_maxrss`, the peak the scorer's lines and its 20 GiB guard use, reached
11.31 GiB.

`chunk-weights.json` gives the candidate's chunk means relative to chunk 1
(1.00, 1.06, 0.98, 0.89, 5.13) and the one-process totals: 62,401 s for the
candidate's 885 slices and 11,690 s for the incumbent's 145, or 4,631 s of
slice work on 16 workers at this speed.

### What this does not establish

- Speed on Modal. The class runs 16 workers on 16 physical cores; per-core
  speed and contention among 16 workers are not measured here. The runbook's
  measured
  Modal slowdown for single-core materialize work (3.3 to 7.6 times per
  chunk) was not measured on this scorer.
- Worker start cost. The pool run below measured it for incumbent workers
  on this host; nothing has measured it on Modal or for the candidate.
- The workers' `gc.collect()` after each slice, which this script does not
  run.
- Why October's sequential run (`h2h-routea-20261009`) took 5.2, 4.7, 6.2,
  5.4 and 10.4 minutes per incumbent slice in chunks 1 to 5: 6 to 9 times the
  incumbent times here in chunks 1 to 4, and 2.7 times in chunk 5. That run
  used an older scorer (checkout a469ebe3d), a free-threaded CPython 3.14t
  environment, and a heap that grew about 0.6 GiB per slice in chunk 5;
  which of these mattered is not tested.
- Within-chunk spread: two slices per chunk, on a loaded host.

## A three-worker pool run

- The pinned scorer itself, run as the d844 stage runs it but with
  `--workers 3`: same inputs, ledger, batch size and one-thread BLAS and
  OpenMP pools, on the same host, shared with other work (its load during
  the run was not recorded). 2026-11-16, launched 19:00:11 UTC.
- Its scorer lines, without paths, are the test fixture
  `packages/microcosm-build/tests/fixtures/us_h2h/pool-run-3-workers-2cfe3043d.log`.
- The incumbent's pool opened at 19:01:14. Its 145th slice ended at
  20:13:52, 4,358 s later, and the parent released the pool at 20:14:04,
  4,370 s after it opened.
- The candidate's pool never opened. While the incumbent's pool ran, the
  candidate H5's folder was moved to an external archive drive at 20:04:13
  (d3015, a stub left at the old path). The scorer opens the candidate only
  after the incumbent's pool, so it stopped at 20:14:04 with
  `FileNotFoundError`. It was not rerun, because the host was above its
  load line for new heavy runs. Both H5s remain in the local Hugging Face
  cache, at the sizes and sha256 values the plan pins.

What the incumbent's pool showed:

- Workers started on demand, one at a time. Each start blocked the parent
  for 42, 36 and 44 s, from the submit that triggered it to the worker's
  first line.
- The estimate's rebuild of the pool from its lines accounts for every
  start, and dates every slice's start before its end (no notes). All three
  starts fall in the opening burst, so this log cannot tell the start rules
  apart: it is consistent with the model, not a test of its steady state.
  The test fixture `toy-pool-12-workers-g44.log`, review r3's toy of the
  pool on CPython 3.14 with twelve workers, has nine starts after a result;
  it is a toy, not the scorer.
- Seconds per slice, excluding workers' first slices: 42.9, 41.8, 41.8,
  43.1 and 262.7 in chunks 1 to 5. The one-process means are 41.6, 46.1,
  42.1, 44.3 and 229.0.
  - Single slices took 36 to 51 s in chunks 1 to 4 and 195 to 338 s in
    chunk 5.
  - Chunk 5 ran 6.1 times chunk 1. One process ran it at 5.5 times chunk 1,
    and the prior scales by that ratio.
  - Chunk 5 slowed within itself: its slices 1 to 12 averaged 227.4 s and
    13 to 29 averaged 287.5 s. Slice 13 started at 19:45:06, about 1,150 s
    before the candidate's folder was moved, so the move does not explain
    it. The cause is not known; host load is a guess.
- Peak RSS, as the scorer prints it (`ru_maxrss` on macOS): 4.38 to
  8.10 GiB in workers' slice lines (their start lines read 1.85 GiB) and
  3.66 to 5.01 GiB in the parent's.
- `scripts/replay_pool_log.py` replays the estimate's model from each 600 s
  poll and compares its predicted last slice with the actual
  (`pool-run-replay.txt`).
  - Every prediction came early, by 259, 246, 263, 436, 216, 32 and 17 s
    (0.4% to 10.0% of the 4,358 s to its last slice).
  - The worst came at 2,400 s, when chunk 5's first measured slices were
    its fastest: a chunk in progress is paced by the slices that have
    ended, and the slow ones were still running.
  - The earlier ones follow the prior's chunk-5 ratio: 236 s per slice
    predicted, 263 s in the pool.
  - On this log the estimate is optimistic. It never overstated the end.

## Simulated runs on Modal (not measurements)

`scripts/projection_grid.py` simulates 16-worker d844 runs with the test
pool simulator (`test_support/microcosm_build/h2h_pool_sim.py`).

- Inputs: the one-process chunk means, times a speed factor f (Modal
  seconds per Mac second), and a worker start cost T.
- It polls the estimate every 600 s against the plan's 14,505 s budget,
  with `chunk-weights.json` as the prior.
- A run fits when its last slice ends at least 600 s before the budget does:
  the estimate's allowance for the comparison and scorecards, which the grid
  also gives them. The incumbent's tail took 12 s on the Mac; the
  candidate's is not timed.
- Output: `projection-grid.txt`; `projection-grid-chunk5-1.5.txt`, with
  chunk 5 a further 1.5 times slower than the prior says for both artifacts;
  `projection-grid-candidate-chunk5-{1.2,1.3,1.5}.txt`, with only the
  candidate's chunk 5 slower; and `projection-grid-pre-tool-300.txt`, with
  `started-at` 300 s before the tool starts.
- The runner's budget counts from the tool's own start, the watcher's from
  `started-at`, which `launch.sh` writes before `modal run`. The other grids
  start both together. Container start and staging (the runbook allows
  300 s) come out of the watcher's budget only, so for the watcher a run
  fits when its last slice ends 600 s plus that gap before the runner's
  budget does.
- Cost is the class's non-preemptible list price, $0.00254688 per second.
- The simulator and the estimate implement the same model, so the grid
  shows only what the watcher would do if Modal behaved like the model. It
  says nothing about Modal's actual speed.

Results:

- With chunk 5 as the prior says, the run fits the budget:
  - f = 1 with T up to 300 s (14,240 s at T = 300);
  - f = 1.5 with T up to 175 s;
  - f = 2 with T up to 100 s;
  - at no T for f of 3 or more.
- Every run that does not fit gets a NO-GO at the first or second estimate,
  600 to 3,000 s in ($1.53 to $7.64). No run that fits gets one; here every
  chunk is the prior times one factor f, so the first estimate's chunk is
  as far from the prior as the rest.
- With chunk 5 1.5 times slower, the fits shrink:
  - f = 1 with T up to 175 s;
  - f = 1.5 with T up to 100 s;
  - f = 2 at T = 40 only.
- There, three runs that do not fit get GO until the incumbent's chunk 5 is
  measured, then NO-GO by 5,400 s ($13.75). Still no run that fits gets a
  NO-GO.
- With only the candidate's chunk 5 slower, nothing before the candidate's
  own chunk 5 shows it. The scorer runs chunk-major, so that chunk, more
  than half the candidate's slice work, comes last. Runs that do not fit
  are caught late or not at all:
  - at 1.2 and 1.3 times, the one run that does not fit only through it
    (f = 1, T = 300 s) gets GO until 12,000 s ($30.56);
  - at 1.5 times, two such runs get NO-GO at 7,800 s ($19.87) and
    12,000 s ($30.56).
  No run that fits gets a NO-GO.
- With `started-at` 300 s before the tool, every estimate moves 300 s
  later, and the one run that fits with less than 900 s to spare (f = 1,
  T = 300 s, 14,240 s) gets a NO-GO at 1,200 s ($3.06). No other row
  changes.

A prior off on one chunk, not all, is what the grid does not cover. The
first estimate measures one chunk and scales its departure from the prior
onto all the work ahead, so it moves by that departure's share of the
scoring still to come. A run that fits with less spare than that can be
stopped at its first estimate, and the stop cannot be undone; a departure
the other way makes the estimate early and a NO-GO late. The test
`test_a_prior_off_on_one_chunk_moves_the_first_estimate_by_its_share` pins
the band: with chunk 1 at 1.1 times the prior, the first estimate is late
by up to a tenth of the scoring after it; a run with less spare than that
error is stopped, and one with a tenth of its run to spare is not.

What this rests on:

- Modal's speed for this scorer. The runbook's measured Modal slowdown for
  single-core materialize work, 3.3 to 7.6 times per chunk, would not fit at
  any start cost. Nothing has measured this scorer on Modal.
- The candidate's start cost. Until its pool logs a start, the estimate uses
  the incumbent's. The candidate frame has 6.1 times the households (177
  slices against 29), so that is likely optimistic, though only until the
  candidate's first start is logged, slice or not.
- The candidate's chunk ratios in a pool. Only the incumbent's were measured
  in one (chunk 5 at 6.1 times chunk 1, against 5.5 in one process); the
  candidate's chunk 5 comes last, so a departure there is seen late.
- Loading the candidate. The estimate leaves it out while the incumbent
  scores; the grid takes it as 11 s times f.
