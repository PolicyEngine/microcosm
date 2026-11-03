"""Pinned MSD oracle through the declared engine gross-up and transfer path."""

import copy
import hashlib
import json
from fractions import Fraction
from random import Random

import pytest

from microcosm.build.transport.compose import load_transport_spec
from test_support.microcosm_build.nz_msd_gross_up import (
    ScheduleFixtureEngine,
    compare_msd_benefit_rates,
    exact_inverse,
    exact_tax,
    schedule_fixture,
)
from tools import build_nz_msd_benefit_rates as generator


def _rates():
    return load_transport_spec("nz")["resources"]["msd_benefit_rates_20260401"]


def test_all_14_msd_points_match_to_the_cent_through_engine_and_annual_transfers():
    engine = ScheduleFixtureEngine()
    comparisons = compare_msd_benefit_rates(engine)
    assert len(comparisons) == 14
    for row in comparisons:
        assert row["model_gross"] == row["before_tax"], row
        assert row["difference"] == "0", row
        exact = exact_inverse(52 * Fraction(row["after_tax_at_m"]), engine.brackets)
        assert row["annual_gross_cents"] == round(exact * 100), row
        annual = Fraction(row["annual_gross_cents"], 100)
        residual = (
            52 * Fraction(row["after_tax_at_m"])
            - annual
            + exact_tax(annual, engine.brackets)
        )
        assert abs(residual) <= Fraction("0.0050001"), row
        assert row["receipt"]["evaluations"] == 48 + 3 + 9
    assert engine.calls == 3 * (48 + 3 + 9)


def test_cent_exact_weekly_nets_clear_certificate_margin_in_every_schedule_bracket():
    """Check all MSD nets plus dense and randomized cent nets using Fractions."""
    brackets = schedule_fixture()
    solves = [
        node
        for node in load_transport_spec("nz")["resources"]["benefit_gross_up"]["nodes"]
        if node["kernel"] == "simulate.solve_zero@1"
    ]
    margin = Fraction("0.000001")
    assert all(
        Fraction(node["params"]["certificate_margin"]) == margin for node in solves
    )
    maximum_gross = Fraction(min(node["params"]["bracket"][1] for node in solves))
    weekly_nets = {
        Fraction(row["after_tax_at_m"])
        for row in _rates()["weekly_main_benefits"]
        if row["individual_validation_point"]
    }
    rng = Random(20250401)
    for index, bracket in enumerate(brackets):
        lower = Fraction(bracket["lower"])
        upper = (
            Fraction(brackets[index + 1]["lower"])
            if index + 1 < len(brackets)
            else maximum_gross
        )
        lower_net_cents = (lower - exact_tax(lower, brackets)) * 100 / 52
        upper_net_cents = (upper - exact_tax(upper, brackets)) * 100 / 52
        first = -(-lower_net_cents.numerator // lower_net_cents.denominator)
        last = upper_net_cents.numerator // upper_net_cents.denominator
        # Dense windows surround both ends of every bracket; integer draws
        # cover the full interval without a float entering the oracle.
        cents = {
            *range(first, min(last + 1, first + 201)),
            *range(max(first, last - 200), last + 1),
            *(rng.randint(first, last) for _ in range(2000)),
        }
        weekly_nets.update(Fraction(value, 100) for value in cents)
    checked_brackets = set()
    minimum = None
    for net in sorted(weekly_nets):
        root = exact_inverse(52 * net, brackets)
        checked_brackets.add(
            max(
                index
                for index, bracket in enumerate(brackets)
                if Fraction(bracket["lower"]) <= root
            )
        )
        nearest_cent = round(root * 100)
        for boundary in (
            Fraction(2 * nearest_cent - 1, 200),
            Fraction(2 * nearest_cent + 1, 200),
        ):
            residual = abs(52 * net + exact_tax(boundary, brackets) - boundary)
            assert residual >= Fraction("0.000025") == 25 * margin, {
                "weekly_net": net,
                "exact_root": root,
                "half_cent_boundary": boundary,
                "absolute_residual": residual,
            }
            minimum = residual if minimum is None else min(minimum, residual)
    assert checked_brackets == set(range(len(brackets)))
    assert len(weekly_nets) >= 10000
    print(
        f"Exact margin check: {len(weekly_nets)} cent-exact weekly nets, "
        f"all 14 MSD rows, {len(checked_brackets)} brackets, min |F(boundary)|={minimum}."
    )


def test_net_as_gross_and_added_acc_levy_cannot_match_the_msd_oracle():
    # An engine returning no tax kills net_as_gross; adding a levy also moves
    # this published comparison. No levy is present in the declared path.
    no_tax = ScheduleFixtureEngine([{"lower": "0", "rate": "0"}])
    assert any(row["difference"] != "0" for row in compare_msd_benefit_rates(no_tax))
    with_levy = copy.deepcopy(schedule_fixture())
    for bracket in with_levy:
        bracket["rate"] = str(float(Fraction(bracket["rate"]) + Fraction("0.0175")))
    assert any(
        row["difference"] != "0"
        for row in compare_msd_benefit_rates(ScheduleFixtureEngine(with_levy))
    )


def test_published_pairs_are_decimal_text_and_super_is_diagnostic_only():
    resource = _rates()
    assert resource["source"] == generator.PIN
    assert len(resource["weekly_main_benefits"]) == 21
    for row in resource["weekly_main_benefits"]:
        assert isinstance(row["after_tax_at_m"], str)
        assert isinstance(row["before_tax"], str)
    assert len(resource["fortnightly_nz_super"]) == 10
    assert all(row["diagnostic_only"] for row in resource["fortnightly_nz_super"])


def test_generator_refuses_a_page_that_does_not_match_the_pinned_receipt(tmp_path):
    receipt = {**generator.PIN, "fetched_at": generator.PIN["retrieved_at"]}
    (tmp_path / "receipts.jsonl").write_text(json.dumps(receipt) + "\n")
    (tmp_path / generator.PAGE).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        generator.generate(tmp_path)
    receipt["sha256"] = hashlib.sha256(b"tampered").hexdigest()
    (tmp_path / "receipts.jsonl").write_text(json.dumps(receipt) + "\n")
    with pytest.raises(ValueError, match="pinned receipt mismatch"):
        generator.generate(tmp_path)


def test_generator_parses_tables_and_check_mode_is_exact(tmp_path, monkeypatch):
    # Small synthetic HTML exercises the real parser, including comma amounts
    # and the N/A S-code hospital cell, without external source dependencies.
    resource = _rates()
    sections = {}
    for row in resource["weekly_main_benefits"]:
        sections.setdefault((row["programme"], row["section"]), []).append(row)
    page = []
    for (programme, section), rows in sections.items():
        heading = next(
            key for key, value in generator.PROGRAMMES.items() if value == programme
        )
        page.append(
            f"<h2>{heading}</h2><h3>{section}</h3><table><tr><th>Header</th></tr>"
        )
        for row in rows:
            category = (
                ""
                if programme == "sole_parent_support"
                else f"<td>{row['category']}</td>"
            )
            page.append(
                f"<tr>{category}<td>${row['after_tax_at_m']}</td><td>${row['before_tax']}</td></tr>"
            )
        page.append("</table>")
    page.append(
        "<h2>NZ Super and Veteran's Pension</h2><table><tr><th>Header</th></tr>"
    )
    for row in resource["fortnightly_nz_super"]:
        at_s = "N/A" if row["after_tax_at_s"] is None else f"${row['after_tax_at_s']}"
        page.append(
            f"<tr><td>{row['category']}</td><td>${row['after_tax_at_m']}</td><td>{at_s}</td><td>${row['before_tax']}</td></tr>"
        )
    page.append("</table>")
    payload = "".join(page).encode()
    pin = {**generator.PIN, "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(generator, "PIN", pin)
    (tmp_path / generator.PAGE).write_bytes(payload)
    (tmp_path / "receipts.jsonl").write_text(
        json.dumps({**pin, "fetched_at": pin["retrieved_at"]}) + "\n"
    )
    output = tmp_path / "rates.json"
    generator.main(["--sources", str(tmp_path), "--out", str(output)])
    generator.main(["--sources", str(tmp_path), "--out", str(output), "--check"])
    assert (
        json.loads(output.read_text())["weekly_main_benefits"]
        == resource["weekly_main_benefits"]
    )
    output.write_text("{}\n")
    with pytest.raises(SystemExit, match="Stale generated"):
        generator.main(["--sources", str(tmp_path), "--out", str(output), "--check"])
