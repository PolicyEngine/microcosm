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
(1.00, 1.06, 0.98, 0.89, 5.13) and the one-process totals, extrapolated
from two timed slices per chunk (each chunk's two-slice mean times its
slice count): 62,401 s for the candidate's 885 slices and 11,690 s for the
incumbent's 145, or 4,631 s of slice work on 16 workers at this speed.

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

- Workers started on demand, one at a time. From the submit that
  triggered each start to the worker's first line took 42, 36 and 44 s,
  intervals rebuilt from the log. How much of each blocked the parent is
  not measured: CPython's `Process.start()` returns once it has written the
  child's arguments down a pipe, and the line comes after the worker's
  initializer.
- The estimate's rebuild of the pool from its lines accounts for every
  start, and dates every slice's start before its end (no notes). All three
  starts fall in the opening burst, so this log cannot tell the start rules
  apart: it is consistent with the model, not a test of its steady state.
  The test fixture `toy-pool-12-workers-g44.log`, review r3's toy of the
  pool on CPython 3.14 with twelve workers, has nine starts after a result,
  four of which run the consume loop's start branch (the rest the opening
  loop's); it is a toy, not the scorer. Its per-process records (float
  clock) are `toy-pool-12-workers-g44-truth.jsonl` beside it, and the test
  checks the rebuild against them.
- Whole-second stamps bias the rebuild slow on the first chunks measured.
  A completion stamped in a spawning submit's second but landing after it
  gives that submit a token, so an idle wait reads as slice time, with no
  note. On the toy, three slices are misdated and chunk 1 reads 6.3% slow
  (chunk 2 5.5%, chunk 3 0.1% fast) against the per-process truth; on
  simulated d844-shaped pools chunk 1 read more than 5% slow in 29 of 200
  (review r4, MODEL-6). The test pins those three slices and those bands.
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
  candidate's is not timed, though the scorer's code suggests it is short.
  So a run whose last slice ends less than 600 s before the budget counts
  as not fitting and is stopped, though it would very likely complete:
  f = 2, T = 175 s (last slice at 13,941 s) gets NO-GO at 1,200 s ($3.06).
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
  as far from the prior as the rest. The grid starts the candidate's
  workers as fast as the incumbent's; the estimate uses the incumbent's
  start cost until the candidate logs one, so a slower candidate start is
  caught about a poll later (review r4: f = 1.5, incumbent starts 100 s,
  candidate 700 s, gets GO at 2,400 and 3,000 s and NO-GO at 3,600 s,
  $9.17).
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

The same band comes back later. Once the incumbent's chunk 5 is measured,
its departure from the prior (1.147 in the real pool) is carried onto the
candidate. Review r4 found that 300 to 600 s late on the grid: f = 1,
T = 300 s (231 s spare) gets NO-GO at 5,400 s ($13.75). Once the candidate
has measured chunks of its own, its unmeasured ones scale by one ratio
pooled over them, carrying the incumbent's departures. A candidate that
does not share them still reads late: by about 800 s in review r4's
scenario with the real pool's departures (the run fitted by 898 s, so
the estimate still passed it, by 105 s). A single slice running far past
its chunk's pace is taken to run as far past it again. That reads early
until the poll passes the midpoint of the pace and the slice's real time,
then late, by up to its real time less the pace as the poll nears its end.
One 3,000 s straggler among 40 s slices reads 1,530 s late polled 2,285 s
in, and up to 2,960 s late just before it ends, so it can stop a run that
fits; the real pool's slowest chunk-5 slice was 1.29 times its mean, which
moves no verdict.

## Lost log lines (simulated, not measurements)

`scripts/lost_lines.py` damages simulated logs the ways a restarted log
stream can, reads the pieces as the watcher's streams, and compares the
projection's verdict and estimate with the intact log's at the same poll.
Output: `lost-lines.txt`; its header gives each section's command. The
property tests draw intact logs only (one stream, or the same log replayed
whole by a second); named examples in the projection's tests and this
offline search cover lost lines.

The cut and boundary searches judge each poll at two budgets per shape:
one 265 s past the true end (the last slice plus the estimate's 600 s
allowance), so the run fits with the spare the d844 shape has at f = 1,
T = 300 s (14,505 s less its 14,240 s true end); and one 600 s short of the
true end, so it overruns. Counts are polls, one per damage case and poll
time. A false NO-GO is a NO-GO from the damaged log, under the fitting
budget, where the intact log gives none. The excess is the damaged
estimate less the intact one at the same poll, where both logs have an
estimate. The search reports it separately for held polls, where the
starts hold applies, and unheld ones (`lost-lines.txt`'s "where the hold
is off"). The starts hold exists because a stretch no stream covers after
a restart (a hole) may contain a pool's worker-start lines: until that
pool logs a start for every worker, the projection turns a time NO-GO (the
estimate past the budget) into WAIT. For the incumbent past its release,
it does so only while the candidate's estimate still uses the incumbent's
start cost, pace, first-slice extra or, for a chunk the candidate has not
measured, departures from the prior. The cut searches poll up to 1,200 s
after the last cut, and the boundary search the next three 600 s polls
after its restart; neither judges a later poll, and a hold can last past
them. At the unheld polls searched, no damage in the search can stop a run
whose intact estimate has more spare than the largest excess; at polls
where only one log has an estimate, or neither, no damaged log gives NO-GO
where the intact log does not, even at the overrunning budget. No damaged
log in the cut or boundary searches gives a stall NO-GO (no scorer line
for an hour, or four times the slowest slice if longer) that the intact
log lacks. The cut and boundary searches project each poll once with no
budget and derive the verdict at each budget from that estimate; the cut
searches also rerun every 50th poll at both budgets, and no rerun verdict
differs from the derived one.

- d844 cuts, 12 shapes: f = 1, 1.01, 1.02 and 1.5 at T = 300 s; f = 1 and 2
  at T = 150 s; f = 3 at T = 100 s; f = 1 at T = 40 and 600 s; and f = 1
  with uneven start costs (100 s for the first start and 320 s for the
  rest, 320 s then 200 s, and 200 s and 400 s alternating). A restart loses
  1, 10 or 60 s at each worker start, pool opening and release line;
  nothing, 10 or 60 s every 120 s; or, in two restarts, 10 s at each of a
  pair of such lines 300 to 2,400 s apart (all such pairs, or an even
  sample of 60 to 90 where a shape has 120 or more). Polls come 5 to
  1,200 s after the last cut. Every shape gives 0 false NO-GOs, and no
  unheld poll reads later than the intact log (largest excess 0 s); held
  polls read up to 3,300 s late. With a 320 s first start cost and 200 s
  for the rest, the intact log itself gives NO-GO at 7,102 polls under the
  fitting budget, which leaves 265 s to spare: the estimate's own error,
  not the damage's.
  The grid above gives every run one start cost, so it does not test this
  shape. Under the overrunning budget, damaged logs give GO where the
  intact log gives NO-GO at 3,004 polls (f = 3, T = 100 s) and 816 (f = 1,
  T = 40 s), so the damage misses the stop at those polls. The search polls
  only to 1,200 s after a cut, so it does not show whether a later poll
  stops those runs; the runner's budget bounds them.
- Random pools drawn as the property tests draw them (seed 3, 100 pools),
  with the same cuts (the regular ones every 300 s). The search runs them
  twice: once with one to four start costs per pool, where the tests draw
  one, and once with each pool's first start cost for every start
  (`--constant-starts`). Both give 0 false NO-GOs and a largest excess of
  0 s at unheld polls; held polls read up to 5,039 s late with varied
  start costs and 5,664 s with constant ones. Before the fix, the starts
  hold (as of aa2addb2d) let false NO-GOs through in pool 94 under
  constant start costs: a lost incumbent start line left the rebuild a
  worker short, so it placed that worker's slices on another, and the
  candidate's estimate, which took that chunk's pace from the misread
  incumbent, read 495 s late once the hold stopped applying at the
  candidate's first start line. The fix (baf2052287) keeps holding a
  released incumbent that is short of starts while the candidate's
  estimate uses any of the incumbent's timing, which closes that case
  (`test_a_lost_incumbent_start_holds_while_the_candidate_paces_from_it`).
  With varied start costs the intact log itself gives NO-GO at 54,346 polls
  of runs that fit by 265 s, and at none with constant ones: the estimate's
  own error, which the search counts separately and does not attribute to
  lost lines. Under the overrunning budget, damaged logs miss the stop at
  4,420 polls (varied) and 7,782 (constant).
- Restarts that lose nothing, across the d844 grid (f = 1 to 3; T = 40 to
  600 s, and 100 s then 320 s): a split every 60 s (in runs that overrun,
  up to the intact log's first NO-GO), polled every 600 s against the
  plan's 14,505 s budget. In the 10 shapes that fit, no split gives NO-GO,
  and 11 to 66 polls per shape (each shape has 105 to 240 splits) turn GO
  into WAIT; the search does not log why, and the code points to waits for
  the new stream to deliver the running slices' lines (inferred). In the
  18 that overrun, the intact log gives NO-GO 600 to 1,800 s in, and every
  split (9 to 29 per shape) gives NO-GO too, all but at most one to three
  of them later: the latest, per shape, 600 to 8,400 s after the intact
  log ($1.53 to $21.39; 8,400 s at T = 600 s for every f). The cause,
  inferred and not traced: a split after the incumbent's pool opens (64 f
  seconds in) leaves a stretch no stream covers, which the projection
  cannot tell from a restart that lost start lines, so the starts hold
  keeps a time NO-GO as WAIT until the incumbent logs its 16th start; a
  split before the pool opens leaves the first stream with no records, so
  no hole forms. The counts fit that exactly in all 18 shapes: the polls
  that turn NO-GO into WAIT equal the splits after the pool opens times the
  largest delay in 600 s polls (at f = 1, T = 450 s, 18 splits times 11
  polls gives 198). One split comes before the pool opens at f = 1 and
  1.5, two at f = 2 and three at f = 3.
  `test_a_lossless_restart_reads_as_the_intact_log_and_holds_only_a_no_go`
  pins the `starts_unconfirmed` flag to the 16th start for one such
  restart, and
  `test_the_hold_reaches_pools_a_hole_could_cut_and_a_stall_still_stops`
  shows a restart that loses nothing holding an estimate past the budget
  as WAIT. In all 18 shapes the delay ends at the first 600 s poll at or
  after the incumbent's 16th start, as arithmetic on the simulator's start
  costs places it (the search does not print start times). The search does
  not cover the grids where a NO-GO comes late anyway (only the candidate's
  chunk 5 slower).
- Restarts inside one second, at every second with records, polled at the
  next three 600 s polls, in the two d844 shapes at f = 1 (T = 300 s, and
  100 s then 320 s) and 30 random pools (seed 3, varied start costs). A new
  stream that replays the whole second, or only the last record seen,
  gives the intact log's verdict at every poll and no later estimate. A
  continuation that splits the second's records between the streams,
  losing nothing, gives 0 false NO-GOs, but the random pools read up to
  24 s late (0 s in both d844 shapes) and one poll reads early enough to
  miss a stop. The streams meet inside the second, so no hole forms and
  the starts hold never applies. Inferred from the code, not traced (the
  boundary command in the header passes no `--rows`, so the output does
  not name the pool or second): the parse counts a line that both streams
  show only as many times as the stream showing it most, so a replay adds
  nothing; but when the split falls between the parent's identical `after
  receiving` lines (one per slice result it takes) in one second, the
  parse keeps only the copies of the stream that shows more of them and
  loses the rest. The
  rebuild then either no longer fits the log (245 polls turn GO into WAIT
  with no estimate, which the code suggests are log gaps) or fits with
  misdated slices. The 24 s is the largest excess at the three polls
  searched after each split, not a bound: a run whose intact estimate had
  less spare than that at such a poll would get a false NO-GO there.
- At the polls searched, the cost is a late or missed stop, not a false
  one, except that a restart splitting one second's lines can read 24 s
  late (above). A lost slice line becomes a log gap once the parent takes
  its result, and a log gap holds WAIT for the rest of the run. A lost
  start holds a time NO-GO as WAIT while its pool is short of starts: the
  candidate's for the rest of its pool, the incumbent's until its release
  and then while the candidate's estimate still uses the incumbent's
  timing. The rebuild shows a log gap once it needs the missing worker,
  which in a pool with fewer slices than two per worker it may never do. A
  restart that loses nothing holds the same way, delaying a NO-GO by up to
  8,400 s ($21.39) here. The runner's budget bounds every case: 14,505 s,
  $36.94 at the class's list price.
- Not covered: a stream that drops lines without ending; Modal's own log
  streams (whether a restart replays has not been tried).

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
