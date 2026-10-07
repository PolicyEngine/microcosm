"""Write the step-1 tables into the README from the published results.

Reads ``results/results.json`` (``tools/run_uk_size_experiment.py publish``) and
replaces the text between the README's results markers. Aggregates only.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
START, END = "<!-- results:start -->", "<!-- results:end -->"


def _pct(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{100 * value:.{digits}f} %"


def _row(label: str, block: dict, dense: dict, acceptance: dict | None) -> str:
    areas = block["areas"]
    criteria = (acceptance or {}).get("criteria", {})
    collapse = sum(
        int(areas[grain].get("below_relative_collapse", 0)) for grain in areas
    )
    return " | ".join(
        [
            label,
            _pct(block["households"] / dense["households"] - 1.0),
            _pct(block["nation_share"].get("Northern Ireland")),
            _pct(block["lone_person_share"]),
            str(areas["constituency"]["below_ess_floor"]),
            str(areas["la"]["below_ess_floor"]),
            str(collapse) if acceptance else "–",
            str(len(block["national_past_25"])),
            f"{block['loss_grain_equal_yardstick']:.4f}",
            ", ".join(sorted(name for name, c in criteria.items() if c.get("pass")))
            or "–",
        ]
    )


def main() -> int:
    results = json.loads((HERE / "results" / "results.json").read_text("utf-8"))
    scorecard = results["scorecard"]
    sets, acceptance = scorecard["sets"], scorecard["acceptance"]
    dense = sets["D"]
    header = (
        "set | households vs D | NI share | lone share | constituencies < 50 | "
        "LAs < 50 | below 25 % of dense ESS | national past 25 % | "
        "grain_equal loss | criteria passed"
    )
    lines = [header, " | ".join(["---"] * 10)]
    lines += [
        _row(label, block, dense, acceptance.get(label))
        for label, block in sets.items()
    ]
    table = "\n".join(f"| {line} |" for line in lines)
    readme = HERE / "README.md"
    text = readme.read_text("utf-8")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    readme.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
