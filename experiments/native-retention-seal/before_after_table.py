"""Emit the before/after table from the three runs' own measurement JSONs.

Reads the native-scale lane's committed baseline and after-run measurements and
this lane's, and prints the comparison the report quotes, so no figure in it is
transcribed by hand.

    python experiments/native-retention-seal/before_after_table.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNS = (
    (
        "baseline 5ff889814",
        ROOT / "experiments/native-scale-transport/measurement-baseline-1-1000.json",
    ),
    (
        "transport after 5307249b3",
        ROOT
        / "experiments/native-scale-transport/measurement-after-1-1000-with-replay.json",
    ),
    (
        "retention seal (this lane)",
        pathlib.Path(
            "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-retention-after"
            "/.measure/after/repeated-verification-measurement.json"
        ),
    ),
)
ROWS = (
    ("runner call CPU s", "runner", "call_cpu_seconds", 2),
    ("runner call wall s", "runner", "call_wall_seconds", 2),
    ("financial node loop wall s (19)", "runner", "node_loop_wall_seconds", 2),
    ("prefix node loop wall s (9)", "runner", "prefix_node_wall_seconds", 2),
    (
        "outside both node loops, wall s",
        "runner",
        "outside_both_node_loops_wall_seconds",
        2,
    ),
    ("whole-process CPU s", "process", "cpu_seconds", 2),
    ("whole-process peak RSS GB", "process", "peak_rss_bytes", None),
)


def main():
    loaded = []
    for label, path in RUNS:
        try:
            loaded.append((label, json.loads(path.read_text()), path))
        except FileNotFoundError:
            loaded.append((label, None, path))
    print("| | " + " | ".join(label for label, _, _ in loaded) + " |")
    print("|---" * (len(loaded) + 1) + "|")
    for name, section, key, places in ROWS:
        cells = []
        for _, document, _ in loaded:
            value = (document or {}).get(section, {}).get(key)
            if value is None:
                cells.append("—")
            elif places is None:
                cells.append(f"{value / 1e9:.2f}")
            else:
                cells.append(f"{value:,.{places}f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    for label, document, _ in loaded:
        print(f"\n{label}: status={(document or {}).get('status')}")


if __name__ == "__main__":
    main()
