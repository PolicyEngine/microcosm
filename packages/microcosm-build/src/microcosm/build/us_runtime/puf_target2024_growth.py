"""Source-specific, explicit PUF2015 to target2024 monetary transport.

This is a data model. It computes no tax, deduction limit, benefit eligibility
or recipient membership. Receipt knownness is separate from raw observation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

# Generated below from the reviewable, source-pinned GROWTH-RECIPE.json.
_RECIPE_JSON = '{\n  "schema": "microcosm.us.puf2015_target2024_growth/1",\n  "input_money_year": 2015,\n  "output_money_year": 2024,\n  "output_profile": "puf59_scf_mortgage_v1",\n  "ordered_targets": [\n    "employment_income_before_lsr",\n    "self_employment_income_before_lsr",\n    "taxable_interest_income",\n    "qualified_dividend_income",\n    "non_qualified_dividend_income",\n    "tax_exempt_interest_income",\n    "short_term_capital_gains",\n    "long_term_capital_gains_before_response",\n    "long_term_capital_gains_on_collectibles",\n    "non_sch_d_capital_gains",\n    "taxable_private_pension_income",\n    "taxable_ira_distributions",\n    "social_security_retirement",\n    "social_security_disability",\n    "social_security_dependents",\n    "social_security_survivors",\n    "alimony_income",\n    "alimony_expense",\n    "salt_refund_income",\n    "charitable_cash_donations",\n    "charitable_non_cash_donations",\n    "real_estate_taxes",\n    "home_mortgage_interest",\n    "investment_interest_expense",\n    "investment_income_elected_form_4952",\n    "student_loan_interest",\n    "educator_expense",\n    "qualified_tuition_expenses",\n    "casualty_loss",\n    "unreimbursed_business_employee_expenses",\n    "traditional_ira_contributions_desired",\n    "self_employed_pension_contributions_desired",\n    "rental_income",\n    "estate_income",\n    "farm_income",\n    "farm_operations_income",\n    "farm_rent_income",\n    "miscellaneous_income",\n    "partnership_income",\n    "s_corp_income",\n    "partnership_self_employment_net_earnings",\n    "estate_income_would_be_qualified",\n    "farm_operations_income_would_be_qualified",\n    "farm_rent_income_would_be_qualified",\n    "partnership_s_corp_income_would_be_qualified",\n    "rental_income_would_be_qualified",\n    "self_employment_income_would_be_qualified",\n    "sstb_self_employment_income_would_be_qualified",\n    "business_is_sstb",\n    "qualified_bdc_income",\n    "qualified_reit_and_ptp_income",\n    "sstb_self_employment_income_before_lsr",\n    "sstb_unadjusted_basis_qualified_property",\n    "sstb_w2_wages_from_qualified_business",\n    "unadjusted_basis_qualified_property",\n    "w2_wages_from_qualified_business",\n    "domestic_production_ald",\n    "unrecaptured_section_1250_gain",\n    "health_savings_account_ald"\n  ],\n  "money_fields": {\n    "employment_income_before_lsr": {\n      "positive_family": "awi",\n      "negative_family": "awi",\n      "positive_factor": "1.452153003110483604210764423019948801037",\n      "negative_factor": "1.452153003110483604210764423019948801037",\n      "assumption": "Average-worker wage index proxy; modeled QBI W2 amounts follow the same wage index."\n    },\n    "w2_wages_from_qualified_business": {\n      "positive_family": "awi",\n      "negative_family": "awi",\n      "positive_factor": "1.452153003110483604210764423019948801037",\n      "negative_factor": "1.452153003110483604210764423019948801037",\n      "assumption": "Average-worker wage index proxy; modeled QBI W2 amounts follow the same wage index."\n    },\n    "sstb_w2_wages_from_qualified_business": {\n      "positive_family": "awi",\n      "negative_family": "awi",\n      "positive_factor": "1.452153003110483604210764423019948801037",\n      "negative_factor": "1.452153003110483604210764423019948801037",\n      "assumption": "Average-worker wage index proxy; modeled QBI W2 amounts follow the same wage index."\n    },\n    "self_employment_income_before_lsr": {\n      "positive_family": "business_profit",\n      "negative_family": "business_loss",\n      "positive_factor": "1.222517420922031096142334753675412822692",\n      "negative_factor": "1.718898116890934040328182671660914217539",\n      "assumption": "Both modeled Schedule C branches use the same sign-specific family factors, preserving the base/SSTB partition."\n    },\n    "sstb_self_employment_income_before_lsr": {\n      "positive_family": "business_profit",\n      "negative_family": "business_loss",\n      "positive_factor": "1.222517420922031096142334753675412822692",\n      "negative_factor": "1.718898116890934040328182671660914217539",\n      "assumption": "Both modeled Schedule C branches use the same sign-specific family factors, preserving the base/SSTB partition."\n    },\n    "taxable_interest_income": {\n      "positive_family": "taxable_interest",\n      "negative_family": "taxable_interest",\n      "positive_factor": "2.599750625636417399355437523686408804619",\n      "negative_factor": "2.599750625636417399355437523686408804619",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "tax_exempt_interest_income": {\n      "positive_family": "tax_exempt_interest",\n      "negative_family": "tax_exempt_interest",\n      "positive_factor": "0.9295395824924337172788943063597248917385",\n      "negative_factor": "0.9295395824924337172788943063597248917385",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "qualified_dividend_income": {\n      "positive_family": "qualified_dividends",\n      "negative_family": "qualified_dividends",\n      "positive_factor": "1.409891045424454325551259680558160347719",\n      "negative_factor": "1.409891045424454325551259680558160347719",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "non_qualified_dividend_income": {\n      "positive_family": "nonqualified_dividends",\n      "negative_family": "nonqualified_dividends",\n      "positive_factor": "2.483795459689869824933006427197633652403",\n      "negative_factor": "2.483795459689869824933006427197633652403",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "non_sch_d_capital_gains": {\n      "positive_family": "non_schedule_d_distributions",\n      "negative_family": "non_schedule_d_distributions",\n      "positive_factor": "1.120350061633613230084419890940102481805",\n      "negative_factor": "1.120350061633613230084419890940102481805",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "taxable_private_pension_income": {\n      "positive_family": "taxable_pensions",\n      "negative_family": "taxable_pensions",\n      "positive_factor": "1.327588034973940867230285994253866499704",\n      "negative_factor": "1.327588034973940867230285994253866499704",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "taxable_ira_distributions": {\n      "positive_family": "ira_distributions",\n      "negative_family": "ira_distributions",\n      "positive_factor": "1.510871073557979130794209158828995676924",\n      "negative_factor": "1.510871073557979130794209158828995676924",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "alimony_income": {\n      "positive_family": "alimony_income",\n      "negative_family": "alimony_income",\n      "positive_factor": "1.542033706590010885068081742929926580179",\n      "negative_factor": "1.542033706590010885068081742929926580179",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "salt_refund_income": {\n      "positive_family": "state_tax_refund",\n      "negative_family": "state_tax_refund",\n      "positive_factor": "0.9296526775720357466732616002244135476521",\n      "negative_factor": "0.9296526775720357466732616002244135476521",\n      "assumption": "Reporting-return mean transport; 2023\\u21922024 stable-real-amount bridge; no unchanged reporting incidence claim."\n    },\n    "short_term_capital_gains": {\n      "positive_family": "short_current_gain",\n      "negative_family": "short_current_loss",\n      "positive_factor": "2.042865587775988756504710527637911842775",\n      "negative_factor": "1.738139193325432816803529458382722438642",\n      "assumption": "Current-year gross gains/loss components per all return proxy for signed net PUF amount; carried losses are excluded."\n    },\n    "long_term_capital_gains_before_response": {\n      "positive_family": "long_current_gain",\n      "negative_family": "long_current_loss",\n      "positive_factor": "1.283710381837894882785950440755768785471",\n      "negative_factor": "2.262384673681395979556694950087192296750",\n      "assumption": "Current-year gross long-term components proxy for signed net amounts and narrower collectibles/1250 subfamilies; carried losses are excluded."\n    },\n    "long_term_capital_gains_on_collectibles": {\n      "positive_family": "long_current_gain",\n      "negative_family": "long_current_loss",\n      "positive_factor": "1.283710381837894882785950440755768785471",\n      "negative_factor": "2.262384673681395979556694950087192296750",\n      "assumption": "Current-year gross long-term components proxy for signed net amounts and narrower collectibles/1250 subfamilies; carried losses are excluded."\n    },\n    "unrecaptured_section_1250_gain": {\n      "positive_family": "long_current_gain",\n      "negative_family": "long_current_loss",\n      "positive_factor": "1.283710381837894882785950440755768785471",\n      "negative_factor": "2.262384673681395979556694950087192296750",\n      "assumption": "Current-year gross long-term components proxy for signed net amounts and narrower collectibles/1250 subfamilies; carried losses are excluded."\n    },\n    "social_security_retirement": {\n      "positive_family": "cola",\n      "negative_family": "cola",\n      "positive_factor": "1.285886314567344447728640",\n      "negative_factor": "1.285886314567344447728640",\n      "assumption": "Cash-year COLA transport of fixed entitlement; carrier component convention remains modeled upstream."\n    },\n    "social_security_disability": {\n      "positive_family": "cola",\n      "negative_family": "cola",\n      "positive_factor": "1.285886314567344447728640",\n      "negative_factor": "1.285886314567344447728640",\n      "assumption": "Cash-year COLA transport of fixed entitlement; carrier component convention remains modeled upstream."\n    },\n    "social_security_dependents": {\n      "positive_family": "cola",\n      "negative_family": "cola",\n      "positive_factor": "1.285886314567344447728640",\n      "negative_factor": "1.285886314567344447728640",\n      "assumption": "Cash-year COLA transport of fixed entitlement; carrier component convention remains modeled upstream."\n    },\n    "social_security_survivors": {\n      "positive_family": "cola",\n      "negative_family": "cola",\n      "positive_factor": "1.285886314567344447728640",\n      "negative_factor": "1.285886314567344447728640",\n      "assumption": "Cash-year COLA transport of fixed entitlement; carrier component convention remains modeled upstream."\n    },\n    "rental_income": {\n      "positive_family": "rental_royalty_profit",\n      "negative_family": "rental_royalty_loss",\n      "positive_factor": "1.476663829685885394438774259073237903799",\n      "negative_factor": "1.690793641559507564235995133214906035185",\n      "assumption": "Related rental/royalty reporting-return mean proxy; published total family includes farm rent and does not exactly equal E25850/E25860 component definitions."\n    },\n    "estate_income": {\n      "positive_family": "estate_profit",\n      "negative_family": "estate_loss",\n      "positive_factor": "1.476331638358104936021886589313386353228",\n      "negative_factor": "1.582810027948011789708036602150526302352",\n      "assumption": ""\n    },\n    "farm_operations_income": {\n      "positive_family": "farm_profit",\n      "negative_family": "farm_loss",\n      "positive_factor": "1.444720208215047581256304812994467826979",\n      "negative_factor": "1.728383897337776476443532509645504817056",\n      "assumption": "Schedule F family; elected farm income T27800 is a selected subfamily proxy, not general farm income."\n    },\n    "farm_income": {\n      "positive_family": "farm_profit",\n      "negative_family": "farm_loss",\n      "positive_factor": "1.444720208215047581256304812994467826979",\n      "negative_factor": "1.728383897337776476443532509645504817056",\n      "assumption": "Schedule F family; elected farm income T27800 is a selected subfamily proxy, not general farm income."\n    },\n    "farm_rent_income": {\n      "positive_family": "farm_rent_profit",\n      "negative_family": "farm_rent_loss",\n      "positive_factor": "1.629452810459537374541656575493532799084",\n      "negative_factor": "1.150378978298943195689490419607276484593",\n      "assumption": ""\n    },\n    "miscellaneous_income": {\n      "positive_family": "other_property_gain",\n      "negative_family": "other_property_loss",\n      "positive_factor": "1.738242164922618767195190248113271033066",\n      "negative_factor": "1.411645744283618865226902643798237455207",\n      "assumption": "Canonical source E01200 is other property gains/losses, not Table1.4 other income."\n    },\n    "partnership_income": {\n      "positive_family": "partnership_s_corp_profit",\n      "negative_family": "partnership_s_corp_loss",\n      "positive_factor": "1.595257264233857504616939193376066035754",\n      "negative_factor": "1.968942227400735507640732147170530268700",\n      "assumption": "Combined component dollars per all return; reporting counts overlap and are never summed. Active partnership earnings use this same explicitly related-family proxy."\n    },\n    "s_corp_income": {\n      "positive_family": "partnership_s_corp_profit",\n      "negative_family": "partnership_s_corp_loss",\n      "positive_factor": "1.595257264233857504616939193376066035754",\n      "negative_factor": "1.968942227400735507640732147170530268700",\n      "assumption": "Combined component dollars per all return; reporting counts overlap and are never summed. Active partnership earnings use this same explicitly related-family proxy."\n    },\n    "partnership_self_employment_net_earnings": {\n      "positive_family": "partnership_s_corp_profit",\n      "negative_family": "partnership_s_corp_loss",\n      "positive_factor": "1.595257264233857504616939193376066035754",\n      "negative_factor": "1.968942227400735507640732147170530268700",\n      "assumption": "Combined component dollars per all return; reporting counts overlap and are never summed. Active partnership earnings use this same explicitly related-family proxy."\n    },\n    "qualified_bdc_income": {\n      "positive_family": "nonqualified_dividends",\n      "negative_family": "nonqualified_dividends",\n      "positive_factor": "2.483795459689869824933006427197633652403",\n      "negative_factor": "2.483795459689869824933006427197633652403",\n      "assumption": "Modeled BDC component follows its parent nonqualified dividend pool; no independent source observation."\n    },\n    "qualified_reit_and_ptp_income": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real proxy for the combined modeled REIT/PTP output, which mixes nonqualified-dividend and pass-through components. No claim that this is a pure dividend pool."\n    },\n    "alimony_expense": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "charitable_cash_donations": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "charitable_non_cash_donations": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "real_estate_taxes": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "home_mortgage_interest": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "investment_interest_expense": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "investment_income_elected_form_4952": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "student_loan_interest": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "educator_expense": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "qualified_tuition_expenses": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "casualty_loss": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "unreimbursed_business_employee_expenses": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "traditional_ira_contributions_desired": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "self_employed_pension_contributions_desired": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "domestic_production_ald": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "health_savings_account_ald": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "unadjusted_basis_qualified_property": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    },\n    "sstb_unadjusted_basis_qualified_property": {\n      "positive_family": "cpi",\n      "negative_family": "cpi",\n      "positive_factor": "1.323487344789614247079323424058189918867",\n      "negative_factor": "1.323487344789614247079323424058189918867",\n      "assumption": "Explicit constant-real-amount proxy. Realized deductions/desires and UBIA are upstream models; this is not a statutory limit, current eligibility, loan turnover, interest-rate, or asset-price model."\n    }\n  },\n  "return_incidence_fields": [\n    "business_is_sstb",\n    "estate_income_would_be_qualified",\n    "farm_operations_income_would_be_qualified",\n    "farm_rent_income_would_be_qualified",\n    "partnership_s_corp_income_would_be_qualified",\n    "rental_income_would_be_qualified",\n    "self_employment_income_would_be_qualified",\n    "sstb_self_employment_income_would_be_qualified"\n  ],\n  "source_pins": {\n    "NATIONAL-GROWTH-EXTRACT.json": "775d89d3fe7a3b1eb9a09c50f6077d41bf838cdb80d9e4aaa9f0ab80ada4e0c8",\n    "INDEX-VALUES.json": "221f3786a462fe105d3da15df3624de5ad3e3f921b4d0205e096067685fd59b3",\n    "extract_national.py": "b6c3a8e4fea52653c01d801d6efe9620cdba4a489d6d7ae8c95cf00ee6dfbe1b",\n    "ACQUISITION.json": "e9a2c34c2ca06a229b2a790ed2987f25d31996653490f342b4b54af37d5f098c"\n  },\n  "sensitivity_cpi_factor": "1.323487344789614247079323424058189918867",\n  "provenance": "All growth applications are modeled distribution transport from observed national series; source observed/derived/modeled flags remain separate.",\n  "operation_order": "Decode source statistical2015 \\u2192 source canonical transformations \\u2192 QBI2015 model (all16 outputs) \\u2192 this growth transform once \\u2192 2024 canonical donor. Monetary recipient predictors remain 2024.",\n  "noninterference": "RECID, S006, raw FLPDYR/FLPDMO, raw money, count source fields and actual survey membership are separate inputs and are not accepted or mutated by this transform.",\n  "sensitivities": "cpi_only uses exactly the same source cohort, zeros, signs and return incidence; no new policy evaluation or calibration authority."\n}\n'
RECIPE_SHA256 = "f08bcc3c37434730bccbf81643c96089383d8e943c60f76ac8337f57a163e1f4"
if hashlib.sha256(_RECIPE_JSON.encode("utf-8")).hexdigest() != RECIPE_SHA256:
    raise ValueError("PUF_GROWTH_RECIPE_IDENTITY")
VERSION = "microcosm.us.puf2015_target2024_growth/1"


def growth_recipe():
    return json.loads(_RECIPE_JSON)


_RECIPE = growth_recipe()
OUTPUTS = tuple(_RECIPE["ordered_targets"])
INCIDENCE_FIELDS = tuple(_RECIPE["return_incidence_fields"])


def _require(value, code):
    if not value:
        raise ValueError(code)


def _digest(columns):
    h = hashlib.sha256()
    for name in OUTPUTS:
        value = columns[name]
        h.update(name.encode("ascii") + b"\0")
        h.update(np.asarray(value, dtype="<f8").tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PufGrowthResult:
    columns: Mapping[str, np.ndarray]
    money_year: int
    receipt: Mapping[str, object]


def grow_puf_2015_to_2024(
    canonical, *, known, input_money_year, scheme="family_observed"
):
    """Grow complete return-grain PUF59 columns once; preserve sign/zero states.

    QBI's basis-sensitive model must run on statistical2015 inputs first.
    Incidence leaves describe zero/one modeled outcomes per return. They are
    returned as integer counts, never grown or called observed person counts.
    Select cpi_only for a same-cohort sensitivity, independently of this call.
    Passing the returned money_year back is rejected, preventing ordinary
    accidental reapplication. The caller owns the source period assertion.
    """
    _require(
        type(input_money_year) is int and input_money_year == 2015,
        "PUF_GROWTH_INPUT_YEAR",
    )
    _require(scheme in ("family_observed", "cpi_only"), "PUF_GROWTH_SCHEME")
    _require(
        isinstance(canonical, Mapping) and isinstance(known, Mapping),
        "PUF_GROWTH_MAPPING",
    )
    _require(
        set(canonical) == set(OUTPUTS) and set(known) == set(OUTPUTS),
        "PUF_GROWTH_ROSTER",
    )
    first = np.asarray(canonical[OUTPUTS[0]])
    _require(first.ndim == 1 and len(first) > 0, "PUF_GROWTH_NONEMPTY_VECTOR")
    n = len(first)
    clean = {}
    for name in OUTPUTS:
        mask = np.asarray(known[name])
        _require(
            mask.shape == (n,) and mask.dtype.kind == "b",
            "PUF_GROWTH_KNOWNNESS_TYPE:" + name,
        )
        _require(bool(mask.all()), "PUF_GROWTH_UNKNOWN:" + name)
        value = np.asarray(canonical[name])
        allowed = "biu" if name in INCIDENCE_FIELDS else "ifu"
        _require(
            value.shape == (n,) and value.dtype.kind in allowed,
            "PUF_GROWTH_PHYSICAL_TYPE:" + name,
        )
        if value.dtype.kind in "iu":
            _require(
                bool(((value >= -(2**53)) & (value <= 2**53)).all()),
                "PUF_GROWTH_INTEGER_PRECISION:" + name,
            )
        _require(bool(np.isfinite(value).all()), "PUF_GROWTH_NONFINITE:" + name)
        if name in INCIDENCE_FIELDS:
            _require(
                bool(((value == 0) | (value == 1)).all()),
                "PUF_GROWTH_INCIDENCE_DOMAIN:" + name,
            )
            clean[name] = value.astype(np.int64)
        else:
            clean[name] = value.astype(np.float64)
    outputs = {}
    for name, value in clean.items():
        if name in INCIDENCE_FIELDS:
            grown = value.copy()
        else:
            rule = _RECIPE["money_fields"][name]
            positive = float(rule["positive_factor"])
            negative = float(rule["negative_factor"])
            if scheme == "cpi_only":
                positive = negative = float(_RECIPE["sensitivity_cpi_factor"])
            _require(
                np.isfinite(positive)
                and np.isfinite(negative)
                and min(positive, negative) > 0,
                "PUF_GROWTH_FACTOR",
            )
            with np.errstate(over="ignore", invalid="ignore"):
                grown = value * np.where(value < 0, negative, positive)
            _require(
                bool(np.isfinite(grown).all()), "PUF_GROWTH_OUTPUT_NONFINITE:" + name
            )
            _require(
                bool(np.array_equal(value == 0, grown == 0))
                and bool(np.array_equal(np.sign(value), np.sign(grown))),
                "PUF_GROWTH_SIGN_OR_ZERO:" + name,
            )
        outputs[name] = np.frombuffer(grown.tobytes(), dtype=grown.dtype)
    receipt = {
        "version": VERSION,
        "recipe_sha256": RECIPE_SHA256,
        "scheme": scheme,
        "input_money_year": 2015,
        "output_money_year": 2024,
        "money_unit": "nominal USD",
        "source_statistical_year": 2015,
        "rows": n,
        "origin": "modeled_distribution_transport",
        "input_knownness": "complete_canonical_after_observed_derived_or_modeled_owner",
        "source_pins": _RECIPE["source_pins"],
        "monetary_columns": 51,
        "unchanged_return_incidence_columns": 8,
        "output_profile": _RECIPE["output_profile"],
        "input_values_sha256": _digest(clean),
        "output_values_sha256": _digest(outputs),
        "no_weight_growth": True,
        "no_extra_raw_row_year_cpi": True,
        "release_eligible": False,
    }
    receipt["sha256"] = hashlib.sha256(
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return PufGrowthResult(MappingProxyType(outputs), 2024, MappingProxyType(receipt))
