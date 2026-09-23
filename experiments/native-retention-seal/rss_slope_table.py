"""The RSS trajectory across the runner call, for the three 1/1000 runs.

The brief asked for the RSS slope across nodes. The harness records each node's
*duration* (``node_wall_times``) and not its absolute start, so nodes cannot be
aligned to the one-second RSS series and a true per-node slope cannot be
fitted from the committed artifacts. What they do support, and what this prints:

* resident bytes at the runner call's entry and at its return, which bracket
  the whole nineteen-node loop and everything around it;
* the growth across that call, and that growth divided by the nineteen nodes --
  an average over the call, explicitly not a fitted slope;
* the peak of the RSS samples that fall inside the call window, which is
  attributable to the call and unlike ``ru_maxrss`` is not a whole-process
  high-water mark.

    python experiments/native-retention-seal/rss_slope_table.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = (
    (
        "baseline 5ff889814",
        "experiments/native-scale-transport/measurement-baseline-1-1000.json",
    ),
    (
        "transport after 5307249b3",
        "experiments/native-scale-transport/measurement-after-1-1000-with-replay.json",
    ),
    (
        "retention seal (this lane)",
        "experiments/native-retention-seal/measurement-after-1-1000-with-replay.json",
    ),
)


def main():
    rows = []
    for label, relative in RUNS:
        document = json.loads((ROOT / relative).read_text())
        runner = document["runner"]
        series = document.get("rss_series") or []
        start, end = runner["call_started_wall_s"], runner["call_returned_wall_s"]
        window = [sample for sample in series if start <= sample[0] <= end]
        entry = runner["call_started_rss_bytes"]
        exit_ = runner["call_returned_rss_bytes"]
        nodes = runner["node_count"]
        rows.append(
            {
                "run": label,
                "entry_gb": entry / 1e9,
                "return_gb": exit_ / 1e9,
                "growth_gb": (exit_ - entry) / 1e9,
                "growth_per_node_mb": (exit_ - entry) / nodes / 1e6,
                "in_call_peak_gb": max(s[1] for s in window) / 1e9 if window else None,
                "in_call_samples": len(window),
                "nodes": nodes,
                "whole_process_peak_gb": document["process"]["peak_rss_bytes"] / 1e9,
            }
        )

    header = (
        "| | entry GB | return GB | growth GB | growth/node MB | in-call peak GB "
        "| ru_maxrss GB |"
    )
    print(header)
    print("|---" * 7 + "|")
    for row in rows:
        print(
            f"| {row['run']} | {row['entry_gb']:.2f} | {row['return_gb']:.2f} | "
            f"{row['growth_gb']:.2f} | {row['growth_per_node_mb']:.0f} | "
            f"{row['in_call_peak_gb']:.2f} | {row['whole_process_peak_gb']:.2f} |"
        )
    print()
    for row in rows:
        print(
            f"{row['run']}: {row['nodes']} nodes, {row['in_call_samples']} "
            "one-second RSS samples inside the runner call"
        )


if __name__ == "__main__":
    main()
