"""Invented source/control coverage; no native files or tax engine."""


def test_puf_target2024_growth_controls():
    import numpy as np

    from microcosm.build.us_runtime import puf_qbi_model as q
    from microcosm.build.us_runtime import puf_target2024_growth as g

    checks = []

    def check(name, value):
        assert bool(value), name
        checks.append(name)

    def fixture(n=128):
        cols = {name: (np.arange(n, dtype=np.float64) + 1) * 100 for name in g.OUTPUTS}
        for name in g.OUTPUTS:
            if name in g.INCIDENCE_FIELDS:
                cols[name] = (np.arange(n) % 2).astype(np.int64)
            else:
                cols[name][::3] *= -1
                cols[name][::7] = 0
        return cols, {k: np.ones(n, dtype=bool) for k in cols}

    def run(cols, known, **kwargs):
        return g.grow_puf_2015_to_2024(
            cols, known=known, input_money_year=2015, **kwargs
        )

    def refuses(label, fn, code):
        try:
            fn()
        except ValueError as e:
            check(label, str(e).startswith(code))
        else:
            raise AssertionError(label + " accepted")

    cols, known = fixture()
    before = {k: v.copy() for k, v in cols.items()}
    result = run(cols, known)
    check(
        "full_59_output_and_price_contract",
        tuple(result.columns) == g.OUTPUTS
        and len(result.columns) == 59
        and result.money_year == 2024,
    )
    check(
        "source_arrays_unchanged",
        all(np.array_equal(v, before[k]) for k, v in cols.items()),
    )
    check(
        "all_signs_and_zero_states_preserved",
        all(
            np.array_equal(np.sign(v), np.sign(cols[k]))
            and np.array_equal(v == 0, cols[k] == 0)
            for k, v in result.columns.items()
        ),
    )
    check(
        "eight_incidence_columns_unchanged_integer",
        all(
            np.array_equal(result.columns[k], cols[k])
            and result.columns[k].dtype.kind == "i"
            for k in g.INCIDENCE_FIELDS
        ),
    )
    check(
        "wage_awi_exact_source_ratio",
        np.allclose(
            result.columns["employment_income_before_lsr"],
            cols["employment_income_before_lsr"] * (69846.57 / 48098.63),
            rtol=1e-15,
            atol=0,
        ),
    )
    cola = np.prod([1, 1.003, 1.02, 1.028, 1.016, 1.013, 1.059, 1.087, 1.032])
    check(
        "ss_cash_year_cola_excludes_2024_december",
        np.allclose(
            result.columns["social_security_retirement"],
            cols["social_security_retirement"] * cola,
            rtol=1e-15,
            atol=0,
        ),
    )
    cpi = run(cols, known, scheme="cpi_only")
    check(
        "same_cohort_cpi_sensitivity",
        all(
            np.allclose(cpi.columns[k], v * (313.689 / 237.017), rtol=1e-15, atol=0)
            for k, v in cols.items()
            if k not in g.INCIDENCE_FIELDS
        ),
    )
    check(
        "scheme_changes_receipt_identity",
        result.receipt["sha256"] != cpi.receipt["sha256"],
    )
    idx = np.arange(127, -1, -1)
    permuted = run(
        {k: v[idx] for k, v in cols.items()}, {k: v[idx] for k, v in known.items()}
    )
    check(
        "row_permutation_replays",
        all(
            np.array_equal(permuted.columns[k], v[idx])
            for k, v in result.columns.items()
        ),
    )
    subset = run(
        {k: v[:13] for k, v in cols.items()}, {k: v[:13] for k, v in known.items()}
    )
    check(
        "subset_replays",
        all(
            np.array_equal(subset.columns[k], v[:13]) for k, v in result.columns.items()
        ),
    )
    refuses(
        "reapplication_year_rejected",
        lambda: g.grow_puf_2015_to_2024(
            result.columns, known=known, input_money_year=result.money_year
        ),
        "PUF_GROWTH_INPUT_YEAR",
    )
    refuses(
        "missing_field_rejected",
        lambda: run(
            {k: v for k, v in cols.items() if k != "taxable_interest_income"}, known
        ),
        "PUF_GROWTH_ROSTER",
    )
    refuses(
        "extra_source_or_mortgage_field_rejected",
        lambda: run({**cols, "RECID": np.arange(128)}, known),
        "PUF_GROWTH_ROSTER",
    )
    for label, value in [
        ("physical_string", np.array(["1"] * 128)),
        ("monetary_bool", np.ones(128, dtype=bool)),
        ("complex", np.ones(128, dtype=complex)),
        ("datetime", np.zeros(128, dtype="datetime64[D]")),
        ("nonfinite", np.full(128, np.nan)),
        ("unsafe_integer", np.full(128, 2**53 + 1, dtype=np.int64)),
        ("wrong_shape", np.zeros((128, 1))),
    ]:
        refuses(
            label,
            lambda value=value: run({**cols, "taxable_interest_income": value}, known),
            "PUF_GROWTH_",
        )
    refuses(
        "unknown_cell_rejected",
        lambda: run(
            cols, {**known, "taxable_interest_income": np.zeros(128, dtype=bool)}
        ),
        "PUF_GROWTH_UNKNOWN:",
    )
    refuses(
        "numeric_knownness_rejected",
        lambda: run(
            cols, {**known, "taxable_interest_income": np.ones(128, dtype=int)}
        ),
        "PUF_GROWTH_KNOWNNESS_TYPE:",
    )
    refuses(
        "incidence_not_a_person_count",
        lambda: run(
            {**cols, "business_is_sstb": np.full(128, 2, dtype=np.int64)}, known
        ),
        "PUF_GROWTH_INCIDENCE_DOMAIN:",
    )
    refuses(
        "monetary_overflow_refused",
        lambda: run(
            {**cols, "taxable_interest_income": np.full(128, np.finfo(float).max)},
            known,
        ),
        "PUF_GROWTH_OUTPUT_NONFINITE:",
    )
    try:
        result.columns["taxable_interest_income"][0] = 2
    except ValueError:
        checks.append("immutable_output_values")
    else:
        raise AssertionError("mutable output")

    # Execute the real archived-assumption QBI model on invented 2015 money.
    base = {k: v.copy() for k, v in cols.items()}
    ids = np.arange(1, 129, dtype=np.int64)
    qbi = q.model_full_puf_qbi(base, ids, known=known, input_money_year=2015, seed=0)
    base.update(qbi.columns)
    after = run(base, known)
    expected = cols["self_employment_income_before_lsr"] * np.where(
        cols["self_employment_income_before_lsr"] < 0,
        float(
            g.growth_recipe()["money_fields"]["self_employment_income_before_lsr"][
                "negative_factor"
            ]
        ),
        float(
            g.growth_recipe()["money_fields"]["self_employment_income_before_lsr"][
                "positive_factor"
            ]
        ),
    )
    check(
        "actual_qbi_schedule_c_partition_conserved_after_growth",
        np.array_equal(
            after.columns["self_employment_income_before_lsr"]
            + after.columns["sstb_self_employment_income_before_lsr"],
            expected,
        ),
    )
    sstb = qbi.columns["business_is_sstb"]
    check(
        "actual_qbi_total_w2_sstb_subpool_preserved",
        np.array_equal(
            after.columns["sstb_w2_wages_from_qualified_business"],
            np.where(sstb, after.columns["w2_wages_from_qualified_business"], 0),
        ),
    )
    check(
        "actual_qbi_total_ubia_sstb_subpool_preserved",
        np.array_equal(
            after.columns["sstb_unadjusted_basis_qualified_property"],
            np.where(sstb, after.columns["unadjusted_basis_qualified_property"], 0),
        ),
    )
    check(
        "qbi_calibration_still_2015",
        qbi.receipt["input_money_year"] == 2015
        and after.receipt["output_money_year"] == 2024,
    )
