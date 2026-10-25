"""UK LCFS consumption imputation stage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import EXPLICIT_KIND, FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    integer_record_keys,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.bus_fare_pricing import (
    PRICE_BUS_JOURNEYS_KIND,
    UK_DFT_BUS_JOURNEYS_RESOURCE,
    bus_fare_prices,
    english_region_boardings,
    price_bus_journeys,
)
from microcosm.build.uk_runtime.donor_uprating import (
    apply_donor_uprating,
    donor_uprating_factors,
    uprating_operation,
)
from microcosm.build.uk_runtime.energy_pricing import (
    CONNECT_RULES,
    DISCONNECT_IDENTITY_UNIFORM_ORDER,
    DISCONNECT_LOWEST_DRAWN_GAS_FIRST,
    ELECTRICITY_KWH,
    GAS_CONNECTED_POSITIVE_SPEND,
    GAS_CONNECTED_PUBLISHED_METER_SHARE,
    GAS_KWH,
    PRICE_DOMESTIC_ENERGY_KIND,
    UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
    UK_NEED_ENERGY_FACTS_RESOURCE,
    UK_QEP_ENERGY_PRICES_RESOURCE,
    EnergyPrices,
    NeedMargins,
    declared_household_kwh_ceilings,
    declared_zero_kwh_cells_at_need_mean,
    impose_gas_connection,
    kwh_to_spend,
    need_margins_from_facts,
    pricing_operation,
    published_energy_level,
    published_gas_connected_shares,
    qep_prices,
    rake_energy_kwh,
    spend_to_kwh,
)
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.gambling_participation import (
    TOP_UP_GAMBLING_PARTICIPATION_KIND,
    UK_GAMBLING_PARTICIPATION_KEYS,
    UK_GAMBLING_PARTICIPATION_RECEIPT,
    GamblingParticipation,
    gambling_participation_operation,
    top_up_gambling_participation,
)
from microcosm.build.uk_runtime.household_income import (
    UK_HBAI_GROSS_INCOME_DEDUCTIONS,
    UK_HBAI_NET_INCOME_PREDICTOR,
    UK_HBAI_NET_INCOME_VARIABLE,
    hbai_gross_income,
    household_level,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows
from microcosm.build.uk_runtime.national_frame import (
    uk_household_mass_conservation_receipt,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    extreme_count,
    floored_share,
    rounded_count,
    small_cell_share,
    support_band,
)
from microcosm.build.uk_runtime.support_clip import (
    UKSupportClipReceipt,
    UKSupportClipResult,
    support_clip_to_donor_with_receipt,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

WEEKS_IN_YEAR = 365.25 / 7

UK_LCFS_CONSUMPTION_DECLARED_SEEDS = {"lcfs_consumption": 0}

LCFS_REGIONS: Mapping[int, str] = {
    1: "NORTH_EAST",
    2: "NORTH_WEST",
    3: "YORKSHIRE",
    4: "EAST_MIDLANDS",
    5: "WEST_MIDLANDS",
    6: "EAST_OF_ENGLAND",
    7: "LONDON",
    8: "SOUTH_EAST",
    9: "SOUTH_WEST",
    10: "WALES",
    11: "SCOTLAND",
    12: "NORTHERN_IRELAND",
}
LCFS_TENURE_MAP: Mapping[int, str] = {
    1: "RENT_FROM_COUNCIL",
    2: "RENT_FROM_HA",
    3: "RENT_PRIVATELY",
    4: "RENT_PRIVATELY",
    5: "OWNED_WITH_MORTGAGE",
    6: "OWNED_WITH_MORTGAGE",
    7: "OWNED_OUTRIGHT",
    8: "RENT_PRIVATELY",
}
LCFS_ACCOMM_MAP: Mapping[int, str] = {
    1: "HOUSE_DETACHED",
    2: "HOUSE_SEMI_DETACHED",
    3: "HOUSE_TERRACED",
    4: "FLAT",
    5: "FLAT",
    6: "MOBILE",
    7: "HOUSE_DETACHED",
    8: "OTHER",
}
#: The donor's annual household weight, thousands of households: LCFS 2024-25
#: (2nd edition) carries it as cal_weight_annual, the name weighta had in
#: earlier years (microcosm#1164). Volume F, sheet "Explanatory Notes", rows
#: 62-64, defines the annual weight as the quarterly weight over four, summing
#: to the UK's households; on the v3 household tab cal_weight_annual is exactly
#: cal_weight_quarterly / 4.
UK_LCFS_DONOR_WEIGHT_COLUMN = "cal_weight_annual"
HOUSEHOLD_LCFS_RENAMES = {
    "g018": "household_adult_count",
    "g019": "household_child_count",
    "gorx": "region",
    "a124": "num_vehicles",
    "p389p": "hbai_household_net_income",
    "p344p": "household_gross_income",
    UK_LCFS_DONOR_WEIGHT_COLUMN: "household_weight",
}
PERSON_LCFS_RENAMES = {
    "b303p": "employment_income",
    "b3262p": "self_employment_income",
    "p049p": "private_pension_income",
}
CONSUMPTION_VARIABLE_RENAMES = {
    "p601": "food_and_non_alcoholic_beverages_consumption",
    "p602": "alcohol_and_tobacco_consumption",
    "p603": "clothing_and_footwear_consumption",
    "p604": "housing_water_and_electricity_consumption",
    "p605": "household_furnishings_consumption",
    "p606": "health_consumption",
    "p607": "transport_consumption",
    "p608": "communication_consumption",
    "p609": "recreation_consumption",
    "p610": "education_consumption",
    "p611": "restaurants_and_hotels_consumption",
    "p612": "miscellaneous_consumption",
    "c72211": "petrol_spending",
    "c72212": "diesel_spending",
    "p537": "domestic_energy_consumption",
}
#: The twelve COICOP division totals (LCFS p601-p612), in division order: the
#: columns policyengine-uk's ``consumption`` sums and its VAT base reads.
UK_LCFS_COICOP_DIVISION_COLUMNS = tuple(
    CONSUMPTION_VARIABLE_RENAMES[f"p6{division:02d}"] for division in range(1, 13)
)
BUS_FARE_LCFS_CODES = ("c73212", "c73213", "c73214")
#: The diary lines of the excise goods inside two COICOP totals the chain
#: draws (microcosm#1359), in the adult forms those totals sum, from the LCF
#: 2024-25 User Guide Volume F (SN 9603,
#: ``9603_volume_f_derived_variables_2024_25.xlsx``), sheet "Part 2". Alcohol
#: brought home, COICOP 02.1 (rows 855-885): spirits and liqueurs, wine,
#: fortified wine, cider and perry, alcopops, champagne and sparkling wine,
#: beer and lager.
UK_LCFS_OFF_TRADE_ALCOHOL_CODES = (
    "c21111",
    "c21211",
    "c21212",
    "c21213",
    "c21214",
    "c21221",
    "c21311",
)
#: Tobacco, COICOP 02.2 (rows 890-900): cigarettes, cigars and other tobacco,
#: whose Volume D products are hand-rolling, pipe and chewing tobacco and
#: e-cigarette refills. P602 sums these ten lines and narcotics (c23111; sheet
#: "Part 3", row 1099), which the end-user tab does not carry.
UK_LCFS_TOBACCO_CODES = ("c22111", "c22121", "c22131")
#: The same drinks away from home, and rounds of drinks: COICOP-plus
#: 11.1.1.1.12 to 11.1.1.1.19 (rows 1749-1770), eight of the lines P611 sums
#: (sheet "Part 3", row 1180).
UK_LCFS_ON_TRADE_ALCOHOL_CODES = (
    "cb111c",
    "cb111d",
    "cb111e",
    "cb111f",
    "cb111g",
    "cb111h",
    "cb111i",
    "cb111j",
)
#: Gambling stakes, COICOP 09.4.3 games of chance (rows 1631-1649): football
#: pools, bingo, lotteries other than the National and Irish ones, bookmaker
#: and tote betting (with gaming machines, which Volume D files there), the
#: Irish Lottery, and National Lottery scratchcards and draws; seven of the
#: lines P609 sums (sheet "Part 3", row 1162). The winnings lines beside
#: them sit in no total.
UK_LCFS_GAMBLING_STAKE_CODES = (
    "c94311",
    "c94312",
    "c94313",
    "c94314",
    "c94315",
    "c94316",
    "c94319",
)
#: The National Lottery's stake lines, scratchcards (row 1646) and draws (row
#: 1649), which Lottery Duty is charged on, and the stake lines beside them.
UK_LCFS_NATIONAL_LOTTERY_STAKE_CODES = ("c94316", "c94319")
UK_LCFS_OTHER_GAMBLING_STAKE_CODES = tuple(
    code
    for code in UK_LCFS_GAMBLING_STAKE_CODES
    if code not in UK_LCFS_NATIONAL_LOTTERY_STAKE_CODES
)
#: Of those, the lines that carry the other duties: football pools (Pool
#: Betting Duty), bingo, and bookmaker, tote and other betting with the gaming
#: machines Volume D files there (rows 1631, 1634 and 1640). Society lotteries
#: (c94313) and the Irish Lottery (c94315) pay no UK gambling duty.
UK_LCFS_BETTING_AND_GAMING_STAKE_CODES = ("c94311", "c94312", "c94314")
#: LCFS fuel, light and power (p537) by fuel, from the LCF 2024-25 User Guide
#: Volume F (SN 9603, ``9603_volume_f_derived_variables_2024_25.xlsx``):
#: the variable labels on sheet "Part 2" (rows cited below) and the P537
#: definition on sheet "Part 3", row 819, the diary's COICOP 04.5 lines (P117,
#: Part 3 row 336) plus the questionnaire's account, meter and combined
#: payments, less the gas and electricity rebates. Weekly amounts
#: (microcosm#1164).
#: Each fuel's last account payment less its rebate (b489 electricity, Part 2
#: row 506; b490 gas, row 507), with the payment and rebate it is derived from
#: (b175/b178, rows 287/289; b170/b173, rows 282/285), which the donor check
#: holds it to (``UK_LCFS_ACCOUNT_IDENTITY_MINIMUM``).
UK_LCFS_ELECTRICITY_ACCOUNT = ("b489", "b175", "b178")
UK_LCFS_GAS_ACCOUNT = ("b490", "b170", "b173")
#: The rest of each fuel: the separate prepayment-meter payment (b1751, b1701;
#: Part 2 rows 288, 283), the fuel's component of a combined utilities bill
#: (b227 electricity, b226 gas; rows 328, 327), its part of a combined
#: key/meter payment (b2332, b2331; rows 336, 335) and the diary's
#: second-dwelling account and slot-meter lines (c45112/c45114, rows
#: 1053/1055; c45212/c45214, rows 1058/1060).
UK_LCFS_ELECTRICITY_PARTS = ("b1751", "b227", "b2332", "c45112", "c45114")
UK_LCFS_GAS_PARTS = ("b1701", "b226", "b2331", "c45212", "c45214")
#: Combined gas-and-electricity payments (b231 bill, Part 2 row 333; b233
#: key/meter, row 334), the rebate netted off each (b174, the combined bill's,
#: row 286; the key/meter payment has none) and the parts the respondent split
#: them into.
UK_LCFS_COMBINED_PAYMENTS: tuple[tuple[str, str | None, tuple[str, str]], ...] = (
    ("b231", "b174", ("b226", "b227")),
    ("b233", None, ("b2331", "b2332")),
)
#: The share of a fuel's account payers whose b489 (b490) must equal b175 less
#: b178 (b170 less b173) to the penny: below it the labels or the rebate's
#: sign have moved, and the stage refuses the donor. On the 2024-25 tab the
#: identity holds for about 98 % of each fuel's account payers.
UK_LCFS_ACCOUNT_IDENTITY_MINIMUM = 0.95
#: The rest of p537 beside electricity and gas: central-heating oil (b017,
#: Part 2 row 211; diary c45312, with paraffin, row 1066), bottled gas (b018,
#: row 212; diary c45222, row 1063), coal and coke (c45411, row 1069), wood and
#: peat (c45412, row 1072), and hot water, steam and ice (c45511, row 1075).
#: The liquid and solid fuels have columns of their own (below); hot water,
#: steam and ice has no engine input and stays in the housing remainder.
UK_LCFS_OTHER_FUEL_PARTS = (
    "b017",
    "b018",
    "c45222",
    "c45312",
    "c45411",
    "c45412",
    "c45511",
)
UK_LCFS_ENERGY_CODES = tuple(
    dict.fromkeys(
        (
            "p537",
            *UK_LCFS_ELECTRICITY_ACCOUNT,
            *UK_LCFS_GAS_ACCOUNT,
            *UK_LCFS_ELECTRICITY_PARTS,
            *UK_LCFS_GAS_PARTS,
            *(
                code
                for total, rebate, parts in UK_LCFS_COMBINED_PAYMENTS
                for code in (total, *(() if rebate is None else (rebate,)), *parts)
            ),
            *UK_LCFS_OTHER_FUEL_PARTS,
        )
    )
)
#: The donor scratch column for the other fuels, read by the split receipt.
UK_LCFS_OTHER_FUELS_COLUMN = "lcfs_other_fuels"
#: The two heating fuels policyengine-uk 2.149.0 reads (pe-uk#2572), parts of
#: housing_water_and_electricity_consumption that add nothing to consumption:
#: liquid fuels (COICOP 04.5.3), central-heating oil and paraffin with bottled
#: gas, LPG with liquid (María's ruling of 2026-10-22); and solid fuels
#: (COICOP 04.5.4), coal, coke, wood and peat. COICOP puts liquefied
#: hydrocarbons under 04.5.2 (gas), and c45222 is a 04.5.2 code, so ONS's 04.5.3
#: leaves out the bottled gas the liquid column holds; the receipt reports it
#: apart (Vahid's review of #1383, item 8). Neither column is levelled to ONS,
#: unlike electricity and gas: the donor's diary totals are about twice ONS's,
#: and the frame's draws reach neither those nor ONS's. No engine formula reads
#: them yet; a costing on either needs levelling to ONS first.
UK_LCFS_FUEL_PARTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "liquid_fuel_consumption": ("b017", "c45312", "b018", "c45222"),
        "solid_fuel_consumption": ("c45411", "c45412"),
    }
)
#: The bottled gas inside the liquid fuels (b018 and its diary line c45222), and
#: the donor scratch column the split receipt reads it from.
UK_LCFS_BOTTLED_GAS_PARTS = ("b018", "c45222")
UK_LCFS_BOTTLED_GAS_COLUMN = "lcfs_bottled_gas"
UK_LCFS_FUEL_COLUMNS = tuple(UK_LCFS_FUEL_PARTS)
#: ``donor.attrs`` key of the donor energy-split receipt.
UK_LCFS_ENERGY_SPLIT_RECEIPT = "donor_energy_split"
#: ``attrs`` key under which the split records the account identity shares.
UK_LCFS_ACCOUNT_IDENTITY_RECEIPT = "account_identity_shares"
#: The donor columns floored at zero before the imputation (microcosm#1063
#: c9): the diary consumption columns and the bus fares summed from the
#: c73212-c73214 diary lines, which a refund can make negative as well.
UK_LCFS_DONOR_FLOORED_COLUMNS = (
    *CONSUMPTION_VARIABLE_RENAMES.values(),
    "bus_fare_spending",
)
#: The spend columns a price index carries from the April-to-March diary to
#: the calendar base year (microcosm#1113). Energy and road fuel are not here:
#: later steps set their levels for that year.
UK_LCFS_DIARY_PRICE_COLUMNS = (
    *UK_LCFS_COICOP_DIVISION_COLUMNS,
    "bus_fare_spending",
    *UK_LCFS_FUEL_COLUMNS,
)
#: Cars and vans available to the household: LCFS ``a124`` on the donor, the
#: was_wealth QRF draw of WAS ``vcarnr8`` on the recipient (the FRS carries no
#: vehicle variable). Both sides are rounded and clipped to this closed range so
#: the consumption QRF sees one predictor scale (microcosm#890 A).
UK_LCFS_VEHICLE_COUNT_RANGE = (0, 5)
#: Vendored Chronicle resources this stage reads; the declaration must name
#: these and no other (the vendor register lists this module as their consumer).
UK_LCFS_ROAD_FUEL_RESOURCE = "road_fuel_anchors.json"
UK_LCFS_LICENSED_CARS_RESOURCE = "licensed_cars_fuel_type.json"
UK_LCFS_DFT_BUS_VALUE_RESOURCE = "dft_bus_value_anchors.json"
# Re-exported for the allowed-resource set; the pricing module is the reader
# the register names.
UK_LCFS_DFT_BUS_JOURNEYS_RESOURCE = UK_DFT_BUS_JOURNEYS_RESOURCE
UK_LCFS_DEVOLVED_BUS_FINANCE_RESOURCE = "devolved_bus_finance.json"
#: ONS Consumer Trends household spending, the road-fuel level (COICOP 07.2.2)
#: and the division totals the capture diagnostic compares against.
UK_LCFS_ONS_EXPENDITURE_RESOURCE = "ons_household_expenditure_facts.json"
UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE = UK_LCFS_ONS_EXPENDITURE_RESOURCE
#: DESNZ sub-national road transport fuel consumption by vehicle and fuel (ktoe),
#: the litres audit's per-fuel cars share.
UK_LCFS_ROAD_FUEL_BY_VEHICLE_RESOURCE = "desnz_road_fuel_by_vehicle.json"
#: The excise clearances the ``level_excise_quantities`` step levels to
#: (microcosm#1359; PolicyEngine/chronicle#381).
UK_LCFS_EXCISE_FACTS_RESOURCE = "excise_facts.json"
UK_LCFS_VENDORED_RESOURCES = (
    UK_LCFS_ROAD_FUEL_RESOURCE,
    UK_LCFS_LICENSED_CARS_RESOURCE,
    UK_LCFS_DFT_BUS_VALUE_RESOURCE,
    UK_LCFS_DFT_BUS_JOURNEYS_RESOURCE,
    UK_LCFS_DEVOLVED_BUS_FINANCE_RESOURCE,
    UK_NEED_ENERGY_FACTS_RESOURCE,
    UK_DESNZ_DOMESTIC_ENERGY_RESOURCE,
    UK_QEP_ENERGY_PRICES_RESOURCE,
    UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE,
    UK_LCFS_ROAD_FUEL_BY_VEHICLE_RESOURCE,
    UK_LCFS_EXCISE_FACTS_RESOURCE,
)
#: The household road-fuel columns the ``level_road_fuel`` step scales.
UK_LCFS_ROAD_FUEL_COLUMNS = ("petrol_spending", "diesel_spending")
#: Columns a declared rake or pricing step levels (energy in kWh to the NEED
#: shape at the DESNZ level; bus fares as journeys times the published
#: yield, microcosm#930; road fuel to ONS household spending, microcosm#1113);
#: the committed support bounds leave them alone.
UK_LCFS_RAKED_COLUMNS = frozenset(
    {
        "electricity_consumption",
        "gas_consumption",
        "domestic_energy_consumption",
        "bus_fare_spending",
        *UK_LCFS_ROAD_FUEL_COLUMNS,
    }
)
LEVEL_ROAD_FUEL_KIND = "level_road_fuel"
LEVEL_EXCISE_QUANTITIES_KIND = "level_excise_quantities"
#: policyengine-uk annualises its weekly tobacco inputs over policyengine-core's
#: 52 weeks (``WEEKS_IN_YEAR``), not the diary's 365.25 / 7.
UK_PE_UK_WEEKS_IN_YEAR = 52
#: The policyengine-uk household inputs the excise step can write
#: (policyengine-uk 2.145.0, ``variables/input/consumption/alcohol`` and
#: ``tobacco``): litres a year of each alcohol product, cigarettes a week and
#: grams a week of the other tobacco products (microcosm#1359).
UK_LCFS_EXCISE_ALCOHOL_OUTPUTS = (
    "beer_litres",
    "cider_litres",
    "sparkling_cider_litres",
    "wine_litres",
    "spirits_litres",
    "other_fermented_litres",
)
UK_LCFS_EXCISE_TOBACCO_OUTPUTS = (
    "cigarettes_per_week",
    "hand_rolling_tobacco_grams_per_week",
    "cigars_grams_per_week",
    "other_tobacco_grams_per_week",
    "heated_tobacco_grams_per_week",
)
#: policyengine-uk's share of each alcohol product that qualifies for draught
#: relief; HMRC publishes draught clearances for beer and cider only, so the
#: others stay at the engine's zero.
UK_LCFS_EXCISE_DRAUGHT_OUTPUTS = (
    "beer_draught_share",
    "cider_draught_share",
)
#: The gambling duty rates the step divides receipts and liabilities by, each
#: the base year's value of the engine parameter the declaration names
#: (policyengine-uk 2.151.0, PolicyEngine/policyengine-uk#2590). The casino
#: rate is the engine's own effective-rate proxy for Gaming Duty.
UK_LCFS_GAMBLING_RATES = (
    "general_betting",
    "pool_betting",
    "remote_gaming",
    "machine_games_lower",
    "machine_games_standard",
    "machine_games_higher",
    "bingo",
    "lottery",
    "casino",
)
#: The gambling inputs each with a duty of its own: the published series its
#: national base comes from, the series' dimension value and the rate that
#: turns that series into the base. HMRC's receipts by duty divided by the
#: rate give the dutiable base the duty was charged on; Machine Games Duty's
#: declared liabilities by rate band divided by the band's rate give each
#: band's net takings; Gaming Duty's receipts divided by the engine's
#: effective-rate proxy give the casino gaming yield that proxy was taken from.
UK_LCFS_GAMBLING_BASES: Mapping[str, tuple[str, str, str]] = {
    "remote_gaming_net_losses": ("receipts", "remote_gaming_duty", "remote_gaming"),
    "pool_betting_net_receipts": ("receipts", "pool_betting_duty", "pool_betting"),
    "bingo_net_receipts": ("receipts", "bingo_duty", "bingo"),
    "national_lottery_ticket_spending": ("receipts", "lottery_duty", "lottery"),
    "land_based_casino_net_losses": ("receipts", "gaming_duty", "casino"),
    "machine_games_lower_rate_net_takings": (
        "machine_games_liabilities",
        "lower_rate",
        "machine_games_lower",
    ),
    "machine_games_standard_rate_net_takings": (
        "machine_games_liabilities",
        "standard_rate",
        "machine_games_standard",
    ),
    "machine_games_higher_rate_net_takings": (
        "machine_games_liabilities",
        "higher_rate",
        "machine_games_higher",
    ),
}
#: The stakes each gambling base is shared out by: Lottery Duty's base by the
#: National Lottery's stakes, every other base by the betting and gaming
#: stakes (football pools, bingo, bookmakers and machines), so neither follows
#: the other's players and the society and Irish lotteries carry no base.
UK_LCFS_NATIONAL_LOTTERY_INPUT = "national_lottery_ticket_spending"
UK_LCFS_NATIONAL_LOTTERY_KEY = "national_lottery"
UK_LCFS_BETTING_AND_GAMING_KEY = "betting_and_gaming"
#: General Betting Duty's base (its receipts over the fixed-odds rate, the
#: rate of every part it is split into in the base year) is split by the
#: Gambling Commission's gross gambling yield for Great Britain: (sector,
#: activity, betting event) cells. Remote fixed-odds betting is split
#: between horse racing and the rest by the horse-racing share of all remote
#: betting yield, since the Commission publishes no activity by event; its
#: horse racing is not only UK racing, which is the part the 2027 remote rate
#: leaves out, and its pool betting can hold pools on other events, which pay
#: Pool Betting Duty. On-course bets are not dutiable, and spread betting,
#: which the Commission does not regulate, has no published base, so both
#: inputs stay at the engine's zero.
UK_LCFS_GENERAL_BETTING_YIELD_CELLS: Mapping[str, tuple[str, str, str]] = {
    "over_the_counter": ("betting_non_remote", "off_course_over_the_counter", "all"),
    "remote_fixed_odds": ("betting_remote", "betting", "all"),
    "exchange": ("betting_remote", "betting_exchange", "all"),
    "pools_non_remote": ("betting_non_remote", "pool_betting", "all"),
    "pools_remote": ("betting_remote", "pool_betting", "all"),
    "remote_horses": ("betting_remote", "all", "horses"),
    "remote_all_events": ("betting_remote", "all", "all"),
}
UK_LCFS_GENERAL_BETTING_OUTPUTS = (
    "land_based_fixed_odds_betting_net_losses",
    "remote_uk_horseracing_betting_net_losses",
    "remote_other_betting_net_losses",
    "betting_exchange_commission",
    "horse_and_dog_pool_betting_net_losses",
)
UK_LCFS_EXCISE_GAMBLING_OUTPUTS = (
    *UK_LCFS_GAMBLING_BASES,
    *UK_LCFS_GENERAL_BETTING_OUTPUTS,
)
UK_LCFS_EXCISE_OUTPUT_COLUMNS = (
    *UK_LCFS_EXCISE_ALCOHOL_OUTPUTS,
    *UK_LCFS_EXCISE_TOBACCO_OUTPUTS,
    *UK_LCFS_EXCISE_DRAUGHT_OUTPUTS,
    *UK_LCFS_EXCISE_GAMBLING_OUTPUTS,
)
#: The units Chronicle stores the clearances in (it applies the publisher's
#: scale, so OBR's thousand hectolitres of alcohol arrive as litres and keep
#: the published unit only as a dimension; PolicyEngine/chronicle#381), as
#: engine units per stored unit: litres (of pure alcohol, which the declared
#: strength turns into litres of the product, or of the product itself),
#: cigarettes, grams. Each row's own unit must be the declared one.
UK_LCFS_EXCISE_ALCOHOL_UNIT = "litre_of_pure_alcohol"
UK_LCFS_EXCISE_UNITS: Mapping[str, float] = {
    UK_LCFS_EXCISE_ALCOHOL_UNIT: 1.0,
    "litre": 1.0,
    "stick": 1.0,
    "kilogram": 1.0e3,
}
#: The unit of the gambling series: pounds a year, the engine's own.
UK_LCFS_EXCISE_GAMBLING_UNIT = "gbp"
#: The units each engine input may be declared in.
UK_LCFS_EXCISE_OUTPUT_UNITS: Mapping[str, tuple[str, ...]] = {
    **{
        column: (UK_LCFS_EXCISE_ALCOHOL_UNIT, "litre")
        for column in UK_LCFS_EXCISE_ALCOHOL_OUTPUTS
    },
    "cigarettes_per_week": ("stick",),
    **{
        column: ("kilogram",)
        for column in UK_LCFS_EXCISE_TOBACCO_OUTPUTS
        if column != "cigarettes_per_week"
    },
}
#: How alcohol's off- and on-trade spending become each household's part of
#: the national volume: in proportion to the two together, as if a pound
#: bought as much alcohol in a pub as in a shop, or each channel's published
#: share of the volume to that channel's spenders.
#:
#: The stage declares spending (María's ruling of 2026-10-23, microcosm#1359).
#: No on-trade share of volume can be archived: MESAS, which publishes one, is
#: a dashboard of NielsenIQ and Circana data with no fixed file. The rule moves
#: no total, since the clearances set each level, but it leans towards
#: drinkers in pubs, where a unit of alcohol costs about three times what it
#: does in a shop. On the LCFS 2024-25 diary the on-trade is about 43 % of
#: alcohol spending, against roughly a quarter of the pure alcohol MESAS
#: counts as sold in England and Wales in 2024. Measured against giving that
#: quarter of the volume to on-trade spenders and the rest to off-trade
#: spenders, a pound spent in a pub carries about 1.6 times its volume and a
#: pound spent in a shop about 0.8 times. About a tenth of the volume, and of
#: the duty, sits with other households; the bottom three income deciles bear
#: about 7 % too little of it and the top decile about 7 % too much, because
#: the on-trade's part of alcohol spending rises with income.
UK_LCFS_ALCOHOL_VOLUME_BY_SPENDING = "spending"
UK_LCFS_ALCOHOL_VOLUME_BY_CHANNEL = "published_on_trade_share"
#: The excise step's receipt key in the stage evidence.
UK_LCFS_EXCISE_QUANTITIES_RECEIPT = "excise_quantities"
#: The LCFS diary lines inside COICOP 07.2.2 beside petrol (c72211) and diesel
#: (c72212): other motor fuels and oils, which ONS's 07.2.2 also carries.
UK_LCFS_OTHER_ROAD_FUEL_CODES = ("c72213",)
UK_LCFS_ROAD_FUEL_OTHER_SHARE_RULE = "lcfs_donor_other_fuels_share"
REDRAW_ZERO_ROAD_FUEL_KIND = "redraw_zero_road_fuel"
UK_LCFS_ROAD_FUEL_REDRAW_RULE = "positive_total_then_petrol_share"
#: The redraw's chain targets: the household's road-fuel total, positive by
#: construction of the training set, then the petrol share of it (zero for a
#: diesel-only household).
UK_LCFS_ROAD_FUEL_TOTAL = "road_fuel_total"
UK_LCFS_PETROL_SHARE = "petrol_share_of_road_fuel"
#: The household input policyengine-uk 2.123.3 reads for the Energy Price
#: Guarantee prepayment discount, July 2023 to March 2024
#: (PolicyEngine/policyengine-uk#2098; microcosm#1110, uk-data#539).
UK_LCFS_PREPAYMENT_METER_COLUMN = "uses_energy_prepayment_meter"
IMPUTE_PREPAYMENT_METER_KIND = "impute_prepayment_meter"
#: What a household last put on a prepayment key or meter (LCF questions
#: ELECMETER, GASMETER and GEMETER, asked only of households that pay by
#: prepayment meter): b1751 electricity, b1701 gas, b233 a combined gas and
#: electricity meter, and the diary's slot-meter lines c45114 (electricity)
#: and c45214 (gas). A household with any of them positive prepays; one with
#: none pays by account or direct debit, or did not say (microcosm#1110).
UK_LCFS_PREPAYMENT_CODES = ("b1751", "b1701", "b233", "c45114", "c45214")
UK_LCFS_PREPAYMENT_RULE = "key_or_meter_payment_positive"
#: Each vehicle is zero-emission at the licensed-car share; the rate is one
#: minus that share (microcosm#1278).
UK_LCFS_ICE_SHARE_RULE = "per_car_zero_emission_share"
#: How the litres audit benchmarks each fuel: HMRC's all-road-user litres times
#: the cars share of that fuel's road use (PolicyEngine/chronicle#322).
UK_LITRES_AUDIT_PER_FUEL_BASIS = "desnz cars share of each fuel's road use"
#: The fallback when DESNZ publishes no vehicle split for the audit's year: the
#: OBR cars share of duty receipts on both fuels, which only the total can bear.
UK_LITRES_AUDIT_UNIFORM_BASIS = "uniform cars share, not a per-fuel benchmark"
UK_DESNZ_ROAD_FUEL_CONCEPT = "desnz.road_transport.fuel_consumption"
UK_LITRES_AUDIT_FUELS = {"petrol_spending": "petrol", "diesel_spending": "diesel"}
#: The litres audit (microcosm#890 C7) reads these vendored concepts beside the
#: declared litre-proxy price concepts: HMRC clearances are all road users, the
#: OBR receipts split names the cars share of them.
UK_HMRC_LITRES_CONCEPTS = {
    "petrol_spending": "hmrc.hydrocarbon_oils.total_petrol_quantity",
    "diesel_spending": "hmrc.hydrocarbon_oils.total_diesel_quantity",
}
UK_OBR_FUEL_DUTY_RECEIPTS_CONCEPT = "obr.fuel_duty.receipts"
UK_OBR_CARS_CATEGORY = "cars"
UK_OBR_TOTAL_CATEGORY = "total"
#: Columns whose level a declared rake sets; the support clip never touches them.
UK_LCFS_DEFAULT_SUPPORT_CLIP_EXEMPT = frozenset(
    {"electricity_consumption", "gas_consumption", "domestic_energy_consumption"}
)
#: The LCFS household's adults (G018) and children (G019), by the age-18 split
#: the survey counts them with. The recipient counts the same split from
#: ``age`` rather than the engine's person flags ``is_adult`` and ``is_child``,
#: which policyengine-uk#1896 deprecates (uk-data#486, microcosm#1095).
UK_LCFS_HOUSEHOLD_COUNT_ADULT_AGE = 18
UK_LCFS_HOUSEHOLD_COUNT_PREDICTORS = ("household_adult_count", "household_child_count")
#: The engine variables the stage materializes on the recipient: the four
#: QRF income predictors and the HBAI deductions that, added back to the HBAI
#: net income, give the recipient's gross income. That only bands the NEED
#: energy rake, on the concept the donor bands on (LCFS p344p; microcosm#1164).
UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS = (
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    UK_HBAI_NET_INCOME_VARIABLE,
    *UK_HBAI_GROSS_INCOME_DEDUCTIONS,
)
UK_LCFS_CONSUMPTION_PREDICTORS = (
    "household_adult_count",
    "household_child_count",
    "region",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "hbai_household_net_income",
    "tenure_type",
    "accommodation_type",
    "num_vehicles",
)
UK_LCFS_CONSUMPTION_TARGET_COLUMNS = (
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
    "petrol_spending",
    "diesel_spending",
    "bus_fare_spending",
    "domestic_energy_consumption",
    "electricity_consumption",
    "gas_consumption",
    # Drawn last, after the totals and the levelled parts they condition on,
    # so every earlier draw is the chain's own (pe-uk#2572).
    "liquid_fuel_consumption",
    "solid_fuel_consumption",
)
#: The shares the chain draws after its spend targets (microcosm#1359). Each
#: splits a COICOP total the chain has drawn, so its parts sum to the drawn
#: total and every total, the VAT base with them, keeps its draw; appended
#: after the spend targets, they move none of those draws either. Tobacco is
#: a share of the off-trade alcohol and tobacco that make up P602, so
#: off-trade alcohol is the rest of that total. On-trade alcohol is a share of
#: restaurants and hotels (P611), whose published total the 2024-25 tab stores
#: to one decimal place, so its lines can pass it by pennies and the share is
#: clipped at one. Gambling stakes are a share of recreation (P609), stored
#: the same way, and are split in turn: the National Lottery's share of the
#: stakes, then the betting and gaming share of the rest, which leaves the
#: society and Irish lotteries no duty base. Each maps to the total it splits
#: (a drawn total, or the stakes an earlier share gives), the diary lines over
#: it and the diary lines (or published total) under it.
UK_LCFS_TOBACCO_SHARE = "tobacco_share_of_alcohol_and_tobacco"
UK_LCFS_ON_TRADE_ALCOHOL_SHARE = "on_trade_alcohol_share_of_restaurants_and_hotels"
UK_LCFS_GAMBLING_SHARE = "gambling_stakes_share_of_recreation"
UK_LCFS_NATIONAL_LOTTERY_SHARE = "national_lottery_share_of_gambling_stakes"
UK_LCFS_BETTING_AND_GAMING_SHARE = "betting_and_gaming_share_of_other_gambling_stakes"
#: The spending the shares give, scratch columns no table carries.
UK_LCFS_TOBACCO_SPENDING = "tobacco_spending"
UK_LCFS_OFF_TRADE_ALCOHOL_SPENDING = "off_trade_alcohol_spending"
UK_LCFS_ON_TRADE_ALCOHOL_SPENDING = "on_trade_alcohol_spending"
UK_LCFS_GAMBLING_SPENDING = "gambling_stakes"
UK_LCFS_NATIONAL_LOTTERY_SPENDING = "national_lottery_stakes"
UK_LCFS_OTHER_GAMBLING_SPENDING = "other_gambling_stakes"
UK_LCFS_BETTING_AND_GAMING_SPENDING = "betting_and_gaming_stakes"
#: The stakes a share can split: scratch, so the donor has no published total
#: of them and the share's denominator is its diary lines.
UK_LCFS_EXCISE_SPENDING_TOTALS = frozenset(
    {UK_LCFS_GAMBLING_SPENDING, UK_LCFS_OTHER_GAMBLING_SPENDING}
)
UK_LCFS_EXCISE_SHARES: Mapping[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    UK_LCFS_TOBACCO_SHARE: (
        "alcohol_and_tobacco_consumption",
        UK_LCFS_TOBACCO_CODES,
        (*UK_LCFS_OFF_TRADE_ALCOHOL_CODES, *UK_LCFS_TOBACCO_CODES),
    ),
    UK_LCFS_ON_TRADE_ALCOHOL_SHARE: (
        "restaurants_and_hotels_consumption",
        UK_LCFS_ON_TRADE_ALCOHOL_CODES,
        ("p611",),
    ),
    UK_LCFS_GAMBLING_SHARE: (
        "recreation_consumption",
        UK_LCFS_GAMBLING_STAKE_CODES,
        ("p609",),
    ),
    UK_LCFS_NATIONAL_LOTTERY_SHARE: (
        UK_LCFS_GAMBLING_SPENDING,
        UK_LCFS_NATIONAL_LOTTERY_STAKE_CODES,
        UK_LCFS_GAMBLING_STAKE_CODES,
    ),
    UK_LCFS_BETTING_AND_GAMING_SHARE: (
        UK_LCFS_OTHER_GAMBLING_SPENDING,
        UK_LCFS_BETTING_AND_GAMING_STAKE_CODES,
        UK_LCFS_OTHER_GAMBLING_STAKE_CODES,
    ),
}
#: ``UKLCFSConsumptionResult`` evidence key of the shares' report-only receipt.
UK_LCFS_EXCISE_SHARES_RECEIPT = "excise_shares"
RECOMPOSE_FROM_REMAINDER_KIND = "recompose_from_remainder"
#: The COICOP totals that contain columns the stage levels (microcosm#1113):
#: each maps to the chain draws subtracted from its own draw to leave its
#: remainder, and the levelled columns added back. The chain draws each part
#: after its total, so the split is the chain's own. Housing (p604) nets the
#: chain's domestic-energy draw, which is the electricity and gas
#: ``derive_energy_from_lcfs`` makes of the diary's fuel, light and power
#: (``p537``) by its labelled parts, so the diary's oil, bottled gas, solid
#: fuels and heat stay in the housing remainder (microcosm#1164); the liquid and
#: solid fuels' own draws are netted and added back unchanged, so the total
#: always holds them (pe-uk#2572). Transport (p607) nets the drawn petrol and
#: diesel (c72211, c72212) and the drawn bus fares (which the pricing step
#: replaces with journeys times yield), and keeps other motor fuels.
UK_LCFS_RECOMPOSED_PARENTS: Mapping[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "housing_water_and_electricity_consumption": (
        ("domestic_energy_consumption", *UK_LCFS_FUEL_COLUMNS),
        ("electricity_consumption", "gas_consumption", *UK_LCFS_FUEL_COLUMNS),
    ),
    "transport_consumption": (
        ("petrol_spending", "diesel_spending", "bus_fare_spending"),
        ("petrol_spending", "diesel_spending", "bus_fare_spending"),
    ),
}
#: The declared ``maximum_total`` of the recomposition (microcosm#1164): each
#: total is held to the donor's realised maximum of that total, the support
#: every drawn column is clipped to, so a remainder near the top of the diary's
#: range plus levelled parts larger than the drawn ones cannot leave the range
#: the terminal support gate holds the emitted totals to.
UK_LCFS_RECOMPOSED_MAXIMUM_TOTAL = "donor_support"
#: The declared levelling of the divisions' relative over-draw (María's rulings
#: of 2026-10-23 and 2026-10-24, microcosm#1278; Vahid's review of #1383, item
#: 9). A division's ratio is the spend the chain drew for it per frame
#: household over the donor's mean of the same spend per donor household: a
#: recomposed total's remainder against the donor's total less the same parts,
#: since its levelled energy, road fuel and priced fares keep their published
#: levels and are no over-draw, and any other division's whole draw. A
#: division above the divisions' shared ratio, measured once before any
#: levelling, comes down to it; no other moves. Measured again after each
#: cut, the shared ratio would fall until every division sat at the lowest
#: one's, close to levelling to the donor totals (María's ruling of
#: 2026-10-24). Only the chain's own draws move.
UK_LCFS_LEVEL_DIVISION_EXCESS = "excess_over_shared_ratio"
UK_LCFS_LEVEL_DIVISION_BASIS = "drawn_spend_over_donor_spend_per_household"
#: The chain draws a recomposition subtracts, kept before any step re-levels them.
UK_LCFS_RECOMPOSED_DRAWN_COLUMNS = tuple(
    dict.fromkeys(
        column
        for subtracted, _ in UK_LCFS_RECOMPOSED_PARENTS.values()
        for column in subtracted
    )
)
#: The excise inputs the stage's declared ``level_excise_quantities`` step
#: writes (microcosm#1359): litres of the five alcohol products HMRC publishes
#: clearances of, the beer and cider draught shares, the four tobacco products
#: and every gambling base. Sparkling cider, heated tobacco and spread betting
#: have no published quantity of their own and keep the engine's zero.
UK_LCFS_EXCISE_DECLARED_OUTPUTS = (
    "beer_litres",
    "cider_litres",
    "wine_litres",
    "spirits_litres",
    "other_fermented_litres",
    *UK_LCFS_EXCISE_DRAUGHT_OUTPUTS,
    "cigarettes_per_week",
    "hand_rolling_tobacco_grams_per_week",
    "cigars_grams_per_week",
    "other_tobacco_grams_per_week",
    *UK_LCFS_EXCISE_GAMBLING_OUTPUTS,
)
UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS = (
    *UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
    "has_fuel_consumption",
    UK_LCFS_PREPAYMENT_METER_COLUMN,
    *UK_LCFS_EXCISE_DECLARED_OUTPUTS,
)
UK_LCFS_CONSUMPTION_NONNEGATIVE_OUTPUT_COLUMNS = UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS
UK_LCFS_CONSUMPTION_FIT_NAME = "uk_lcfs_2024_25_consumption"
UK_LCFS_CONSUMPTION_STAGE_NAME = "lcfs_consumption"


@dataclass
class UKLCFSConsumptionResult:
    """Transformed frame and donor-support clip receipt."""

    frame: Frame
    support_clip: UKSupportClipReceipt
    donor_uprating: Mapping[str, Any] | None = None
    fuel_flag: Mapping[str, Any] | None = None
    prepayment_meter: Mapping[str, Any] | None = None
    donor_energy_split: Mapping[str, Any] | None = None
    bus_pricing: Mapping[str, Any] | None = None
    energy_pricing: Mapping[str, Any] | None = None
    energy_rake: Mapping[str, Any] | None = None
    fuel_litres_audit: Mapping[str, Any] | None = None
    donor_floor: Mapping[str, Any] | None = None
    road_fuel_incidence: Mapping[str, Any] | None = None
    road_fuel_level: Mapping[str, Any] | None = None
    recomposed_totals: Mapping[str, Any] | None = None
    division_capture: Mapping[str, Any] | None = None
    excise_shares: Mapping[str, Any] | None = None
    gambling_participation: Mapping[str, Any] | None = None
    excise_quantities: Mapping[str, Any] | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
            "stage": UK_LCFS_CONSUMPTION_STAGE_NAME,
            "support_clip": self.support_clip.evidence(),
        }
        if self.donor_uprating is not None:
            evidence["donor_uprating"] = dict(self.donor_uprating)
        if self.fuel_flag is not None:
            evidence["has_fuel_consumption"] = dict(self.fuel_flag)
        if self.prepayment_meter is not None:
            evidence[UK_LCFS_PREPAYMENT_METER_COLUMN] = dict(self.prepayment_meter)
        if self.donor_energy_split is not None:
            evidence[UK_LCFS_ENERGY_SPLIT_RECEIPT] = dict(self.donor_energy_split)
        if self.bus_pricing is not None:
            evidence["bus_pricing"] = dict(self.bus_pricing)
        if self.energy_pricing is not None:
            evidence["energy_pricing"] = dict(self.energy_pricing)
        if self.energy_rake is not None:
            evidence["energy_rake"] = dict(self.energy_rake)
        if self.fuel_litres_audit is not None:
            evidence["fuel_litres_audit"] = dict(self.fuel_litres_audit)
        if self.donor_floor is not None:
            evidence["donor_floor"] = dict(self.donor_floor)
        if self.road_fuel_incidence is not None:
            evidence["road_fuel_incidence"] = dict(self.road_fuel_incidence)
        if self.road_fuel_level is not None:
            evidence["road_fuel_level"] = dict(self.road_fuel_level)
        if self.recomposed_totals is not None:
            evidence["recomposed_totals"] = dict(self.recomposed_totals)
        if self.division_capture is not None:
            evidence["division_capture"] = dict(self.division_capture)
        if self.excise_shares is not None:
            evidence[UK_LCFS_EXCISE_SHARES_RECEIPT] = dict(self.excise_shares)
        if self.gambling_participation is not None:
            evidence[UK_GAMBLING_PARTICIPATION_RECEIPT] = dict(
                self.gambling_participation
            )
        if self.excise_quantities is not None:
            evidence[UK_LCFS_EXCISE_QUANTITIES_RECEIPT] = dict(self.excise_quantities)
        return evidence


#: The household-mass receipt this stage records (the manifest's
#: ``record_mass_conservation_receipt`` operation repeats it): the terminal
#: family gate requires exactly this reason on a valid mass-conserving record.
UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON = "LCFS consumption imputation on the source spine: household weights pass through unchanged and total household mass is conserved."


@dataclass
class UKLCFSConsumptionStageTransform:
    """Whole-stage callable for LCFS-trained consumption imputation."""

    stage: SourceStageSpec
    engine: object
    lcfs_hh_tab_path: str | Path | None = None
    lcfs_person_tab_path: str | Path | None = None
    lcfs_household: pd.DataFrame | None = None
    lcfs_person: pd.DataFrame | None = None
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    last_result: UKLCFSConsumptionResult | None = field(default=None, init=False)

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        if self.last_fit_weight_records is None:
            return ()
        return tuple(self.last_fit_weight_records)

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        lcfs_household = (
            self.lcfs_household
            if self.lcfs_household is not None
            else read_pinned_tab(
                _require_path(self.lcfs_hh_tab_path),
                _artifact(self.stage, "lcfs_household_tab"),
            )
        )
        lcfs_person = (
            self.lcfs_person
            if self.lcfs_person is not None
            else read_pinned_tab(
                _require_path(self.lcfs_person_tab_path),
                _artifact(self.stage, "lcfs_person_tab"),
            )
        )
        uprating_factors, uprating_receipt = lcfs_donor_uprating(self.stage)
        energy = lcfs_energy_pricing(self.stage)
        shares = lcfs_donor_shares(self.stage)
        donor = clean_lcfs_consumption_table(
            lcfs_person,
            lcfs_household,
            uprating=uprating_factors,
            energy=energy,
            donor_rake_iterations=_donor_energy_rake_iterations(self.stage),
            shares=shares,
        )
        require_refused_unmapped_codes(self.stage)
        donor, donor_floor_receipt = floor_negative_donor_consumption(
            donor, floor=donor_consumption_floor(self.stage)
        )
        ice_share, ice_share_receipt = lcfs_ice_share(self.stage)
        recipient = recipient_predictors(frame, self.engine)
        recipient["has_fuel_consumption"] = assign_recipient_has_fuel(
            frame,
            rate=ice_share,
            seed=_operation_seed(self.stage, "assign_binary_from_rate"),
        )
        weights = frame.weights_for("household").values
        # The receipts' floors count source households, not the copies the
        # frame stacks of one (microcosm#1306 review).
        sources = _source_households(frame.table("household"))
        fuel_flag_receipt = fuel_flag_evidence(
            donor,
            recipient,
            recipient_weights=weights,
            ice_share_receipt=ice_share_receipt,
            sources=sources,
        )
        imputation = impute_lcfs_consumption(
            donor,
            recipient,
            seed=_operation_seed(self.stage, "fit_weighted_qrf_chain"),
            n_estimators=_qrf_n_estimators(self.stage),
            record_ids=frame.table("household")["household_id"].to_numpy(),
            targets=lcfs_chain_targets(self.stage, shares),
        )
        clip_result = support_clip_to_donor(
            imputation.draws, donor, exempt=support_clip_exempt(self.stage)
        )
        household_draws, shares_outside = clip_excise_shares(
            clip_result.clipped, shares
        )
        prepayment = lcfs_prepayment_meter(
            self.stage,
            donor=donor,
            recipient=recipient,
            household_ids=frame.table("household")["household_id"].to_numpy(),
            weights=weights,
        )
        # The chain's own draws of the parts later steps re-level, for the
        # recomposed totals (microcosm#1113).
        drawn_parts = household_draws[list(UK_LCFS_RECOMPOSED_DRAWN_COLUMNS)].copy()
        energy_rake_receipt = None
        gas_payers = None
        if energy is not None and energy.connect_rule is not None:
            gas_payers = gas_payer_draws(
                donor=donor,
                recipient=recipient,
                household_ids=frame.table("household")["household_id"].to_numpy(),
                energy=energy,
            )
        if energy is not None:
            household_draws, energy_rake_receipt = rake_recipient_energy(
                household_draws,
                energy=energy,
                region=recipient["region"].astype(str).to_numpy(),
                income=recipient["household_gross_income"].to_numpy(dtype=float),
                tenure=recipient["tenure_type"].astype(str).to_numpy(),
                accommodation=recipient["accommodation_type"].astype(str).to_numpy(),
                weights=weights,
                iterations=_recipient_energy_rake_iterations(self.stage),
                identity=frame.table("household")["household_id"].to_numpy(),
                gas_payers=gas_payers,
            )
        household_draws, bus_pricing_receipt = lcfs_bus_fare_pricing(
            self.stage, household_draws, frame
        )
        household_draws["domestic_energy_consumption"] = (
            household_draws["electricity_consumption"]
            + household_draws["gas_consumption"]
        )
        household_draws.loc[
            ~recipient["has_fuel_consumption"].astype(bool),
            ["petrol_spending", "diesel_spending"],
        ] = 0.0
        incidence = lcfs_road_fuel_incidence(
            self.stage,
            household_draws,
            donor=donor,
            recipient=recipient,
            household_ids=frame.table("household")["household_id"].to_numpy(),
            weights=weights,
        )
        household_draws = incidence.draws
        household_draws, road_fuel_level_receipt = lcfs_road_fuel_level(
            self.stage, household_draws, weights=weights, lcfs_household=lcfs_household
        )
        household_draws, recomposed_totals_receipt = lcfs_recompose_from_remainder(
            self.stage,
            household_draws,
            drawn=drawn_parts,
            weights=weights,
            donor_support=donor_realized_ranges(donor),
            uprating_factors=uprating_factors,
            donor_division_means=lcfs_donor_drawn_means(donor),
        )
        energy_split = donor.attrs.get(UK_LCFS_ENERGY_SPLIT_RECEIPT)
        if recomposed_totals_receipt is not None and isinstance(energy_split, Mapping):
            recomposed_totals_receipt["published_fuels"] = published_fuels_beside_draws(
                recomposed_totals_receipt["published_fuels"],
                energy_split,
                {
                    column: float(np.dot(weights, household_draws[column].to_numpy()))
                    for column in UK_LCFS_FUEL_COLUMNS
                },
            )
        uprating = uprating_operation(self.stage)
        capture = (
            None
            if uprating is None
            else division_capture_or_reason(
                household_draws,
                weights=weights,
                period_value=int(uprating["to_period"]),
                donor=donor,
                sources=sources,
            )
        )
        litres_audit = fuel_litres_audit(
            household_draws, weights=weights, stage=self.stage
        )
        excise_shares = (
            excise_shares_receipt(
                household_draws,
                shares=shares,
                weights=weights,
                donor_receipt=donor.attrs[UK_LCFS_EXCISE_SHARES_RECEIPT],
                outside=shares_outside,
            )
            if shares
            else None
        )
        gambling = lcfs_gambling_participation(
            self.stage,
            household_draws,
            donor=donor,
            recipient=recipient,
            drawn=imputation.draws,
            person=frame.table("person"),
            household_ids=frame.table("household")["household_id"].to_numpy(),
            weights=weights,
            sources=sources,
            shares=shares,
        )
        quantities, excise_quantities = lcfs_excise_quantities(
            self.stage,
            household_draws,
            weights=weights,
            shares=shares,
            participation=None if gambling is None else gambling.keys,
        )
        household = frame.table("household").copy()
        for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
            household[column] = household_draws[column].to_numpy()
        household["has_fuel_consumption"] = recipient["has_fuel_consumption"].to_numpy(
            dtype=bool
        )
        if prepayment is not None:
            household[UK_LCFS_PREPAYMENT_METER_COLUMN] = prepayment.flags
        if quantities is not None:
            # In the declared output order, which the stage's outputs and the
            # graph's typed cells follow; an undeclared input fails here.
            for column in sorted(
                quantities.columns, key=UK_LCFS_EXCISE_DECLARED_OUTPUTS.index
            ):
                household[column] = quantities[column].to_numpy(dtype=float)
        result = uk_national_frame(
            person=frame.table("person").copy(),
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=(
                *frame.mass_log,
                uk_household_mass_conservation_receipt(
                    frame, UK_LCFS_CONSUMPTION_MASS_CONSERVATION_REASON
                ),
            ),
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = (
            *imputation.fit_weight_records,
            *incidence.fit_weight_records,
            *(() if prepayment is None else prepayment.fit_weight_records),
            *(() if gas_payers is None else gas_payers.fit_weight_records),
            *(() if gambling is None else gambling.fit_weight_records),
        )
        self.last_result = UKLCFSConsumptionResult(
            frame=result,
            support_clip=clip_result.receipt,
            donor_uprating=uprating_receipt,
            fuel_flag=fuel_flag_receipt,
            prepayment_meter=None if prepayment is None else prepayment.receipt,
            donor_energy_split=donor.attrs.get(UK_LCFS_ENERGY_SPLIT_RECEIPT),
            bus_pricing=bus_pricing_receipt,
            energy_pricing=None if energy is None else energy.receipt,
            energy_rake=energy_rake_receipt,
            fuel_litres_audit=litres_audit,
            donor_floor=donor_floor_receipt,
            road_fuel_incidence=incidence.receipt,
            road_fuel_level=road_fuel_level_receipt,
            recomposed_totals=recomposed_totals_receipt,
            division_capture=capture,
            excise_shares=excise_shares,
            gambling_participation=None if gambling is None else gambling.receipt,
            excise_quantities=excise_quantities,
        )
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_LCFS_CONSUMPTION_OUTPUT_COLUMNS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


@dataclass(frozen=True)
class UKLCFSConsumptionImputationResult:
    draws: pd.DataFrame
    fit_weight_records: tuple[FitWeightRecord, ...]


class DiagnosticUnavailableError(ValueError):
    """A report-only diagnostic's published row is missing for the year.

    Raised only by the diagnostics' own row lookups when the vendored resource
    carries no row for the period, so the stage can record why a report-only
    block has no figures and carry on. An ambiguous or duplicate row is a
    data-integrity fault and fails closed as a plain ``ValueError``; a
    vendored resource that lags the feed pin still fails in ``vendored_rows``
    (microcosm#1113, the #1167 re-review).
    """


#: ONS Consumer Trends' generic concept for the divisions and classes.
UK_ONS_HFCE_CONCEPT = "ons.household_final_consumption_expenditure"
#: The variant of each Consumer Trends row the diagnostic reads: annual,
#: current prices, not seasonally adjusted.
UK_ONS_HFCE_ROW_DIMENSIONS = {
    "frequency": "annual",
    "price_basis": "current_prices",
    "seasonal_adjustment": "not_seasonally_adjusted",
}
#: Each LCFS division column and its COICOP division.
UK_LCFS_DIVISION_COICOP = {
    column: f"{division:02d}"
    for division, column in enumerate(UK_LCFS_COICOP_DIVISION_COLUMNS, start=1)
}
#: The classes national accounts count that a household diary does not:
#: narcotics, owner-occupiers' imputed rent and FISIM, each with its sheet.
UK_LCFS_SURVEY_OUT_OF_SCOPE_CLASSES = {
    "02": (("02.3", "02CN"),),
    "04": (("04.2", "04CN"),),
    "12": (("12.6.1", "12CN"),),
}


def _ons_annual_value(
    coicop: str, *, source_sheet: str, consumption_concept: str, period_value: int
) -> tuple[float, str]:
    rows = vendored_rows(
        UK_LCFS_ONS_EXPENDITURE_RESOURCE,
        concept=UK_ONS_HFCE_CONCEPT,
        period_type="calendar_year",
        period_value=int(period_value),
        dimensions={
            **UK_ONS_HFCE_ROW_DIMENSIONS,
            "coicop": coicop,
            "source_sheet": source_sheet,
            "consumption_concept": consumption_concept,
        },
    )
    if not rows:
        raise DiagnosticUnavailableError(
            f"{UK_LCFS_ONS_EXPENDITURE_RESOURCE}: no {coicop} "
            f"({source_sheet}, {consumption_concept}) row for {period_value}."
        )
    if len(rows) != 1:
        raise ValueError(
            f"{UK_LCFS_ONS_EXPENDITURE_RESOURCE}: expected one {coicop} "
            f"({source_sheet}, {consumption_concept}) row for {period_value}, "
            f"found {len(rows)}."
        )
    return float(rows[0]["value"]), str(rows[0].get("source_record_id", ""))


#: The top shares of each division's spend the capture receipt reports, on
#: the frame and on its donor (microcosm#1278: whether a division's over-draw
#: sits in the tail).
UK_LCFS_DIVISION_TAIL_FRACTIONS = (0.01, 0.10)


def weighted_top_shares(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    fractions: Sequence[float],
    sources: Sequence[Any] | None = None,
) -> dict[float, float | str | None]:
    """The share of the weighted total held by the top spenders, per fraction.

    For each fraction the highest-spending households are taken in order
    until they carry that fraction of the weight, so a smaller fraction's
    households lead a larger one's. The floors count the distinct sources
    that spend anything (``sources``, one per row when omitted: a frame
    stacks copies of one source household), as every receipt count is (UKDS
    disclosure control), and a share is ``"<10"`` when one to nine of them:
    - sit in its top group;
    - sit outside it, since the receipt's total less the top group's spend
      gives back theirs;
    - or sit between it and the last shipped smaller fraction's group, since
      the two shares' difference gives back their spend (microcosm#1306
      review).

    ``None`` when the total is not positive.
    """

    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"a top fraction must lie in (0, 1]: {fraction}.")
    value = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    source = (
        np.arange(len(value)) if sources is None else np.asarray(sources, dtype=object)
    )
    if len(source) != len(value):
        raise ValueError("sources must align with the values.")
    positive = weight > 0
    value, weight, source = value[positive], weight[positive], source[positive]
    total = float(np.dot(weight, value))
    if not len(value) or total <= 0:
        return {fraction: None for fraction in fractions}
    order = np.argsort(-value, kind="stable")
    cumulative = np.cumsum(weight[order])

    def spenders(rows: np.ndarray) -> int:
        return len(set(source[rows][value[rows] != 0].tolist()))

    shares: dict[float, float | str | None] = {}
    shipped_rows: int | None = None
    for fraction in sorted(fractions):
        cut = fraction * cumulative[-1]
        rows = int(np.searchsorted(cumulative, cut, side="left")) + 1
        top = order[:rows]
        share: float | str | None = float(np.dot(weight[top], value[top]) / total)
        for group in (
            top,
            order[rows:],
            order[shipped_rows:rows] if shipped_rows is not None else order[:0],
        ):
            share = small_cell_share(share, spenders(group))
            if not isinstance(share, float):
                break
        if isinstance(share, float):
            shipped_rows = rows
        shares[fraction] = share
    return shares


def weighted_top_share(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    fraction: float,
    sources: Sequence[Any] | None = None,
) -> float | str | None:
    """One fraction's :func:`weighted_top_shares`."""

    return weighted_top_shares(values, weights, fractions=(fraction,), sources=sources)[
        fraction
    ]


def division_tail(
    frame_values: Sequence[float],
    frame_weights: Sequence[float],
    donor_values: Sequence[float],
    donor_weights: Sequence[float],
    *,
    frame_sources: Sequence[Any] | None = None,
) -> dict[str, dict[str, float | str | None]]:
    """Top-1% and top-10% spend shares on the frame and its donor, and their ratio.

    The frame's floor counts its distinct source households
    (``frame_sources``); the donor has one row per LCFS household.
    """

    frame_shares = weighted_top_shares(
        frame_values,
        frame_weights,
        fractions=UK_LCFS_DIVISION_TAIL_FRACTIONS,
        sources=frame_sources,
    )
    donor_shares = weighted_top_shares(
        donor_values, donor_weights, fractions=UK_LCFS_DIVISION_TAIL_FRACTIONS
    )
    tail: dict[str, dict[str, float | str | None]] = {}
    for fraction in UK_LCFS_DIVISION_TAIL_FRACTIONS:
        frame_share, donor_share = frame_shares[fraction], donor_shares[fraction]
        both = isinstance(frame_share, float) and isinstance(donor_share, float)
        tail[f"top_{round(fraction * 100)}_percent"] = {
            "frame": frame_share,
            "donor": donor_share,
            "frame_over_donor": (
                frame_share / donor_share if both and donor_share > 0 else None
            ),
        }
    return tail


def division_capture(
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    period_value: int,
    donor: pd.DataFrame | None = None,
    sources: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """How much of ONS household spending each LCFS division captures.

    Diagnostic only (microcosm#1113): each division's prior-weighted frame
    total against ONS Consumer Trends for the same calendar year, less the
    classes a household diary does not record (narcotics, imputed rent,
    FISIM). The divisions are ONS's domestic concept: they include
    non-residents' spending in the UK and exclude residents' spending abroad,
    and the tourism adjustment is published for the total only, so only the
    total is also compared on the national concept. Nothing is gated on it.

    With ``donor`` (the annualised donor the chain drew from, carried to the
    base year), each division and the total also record the donor's weighted
    total and the frame's over it, in all and per household: whether the chain
    draws a division above its own donor, as microcosm#1167 found it did for
    the transport remainder (microcosm#1207's furnishings and health item).
    Each division also records the share of its spend held by the top 1% and
    10% of households on both sides (:func:`division_tail`), so an over-draw
    can be placed in the tail or the body (microcosm#1278); ``sources`` names
    each frame row's source household, so the frame's floor counts distinct
    sources rather than stacked copies.
    """

    weight = np.asarray(weights, dtype=float)
    donor_weight = (
        None
        if donor is None
        else _numeric(donor["household_weight"]).to_numpy(dtype=float)
    )
    frame_households = float(weight.sum())
    donor_households = None if donor_weight is None else float(donor_weight.sum())

    def beside_donor(frame: float, donor_total: float) -> dict[str, Any]:
        usable = donor_total > 0 and frame_households > 0 and bool(donor_households)
        return {
            "donor": donor_total,
            "frame_over_donor": frame / donor_total if donor_total > 0 else None,
            "frame_over_donor_per_household": (
                (frame / frame_households) / (donor_total / donor_households)
                if usable
                else None
            ),
        }

    divisions: dict[str, Any] = {}
    frame_total = 0.0
    donor_grand_total = 0.0
    out_of_scope_total = 0.0
    for column, coicop in UK_LCFS_DIVISION_COICOP.items():
        published, record_id = _ons_annual_value(
            coicop,
            source_sheet="0CN",
            consumption_concept="domestic",
            period_value=period_value,
        )
        out_of_scope = {}
        for code, sheet in UK_LCFS_SURVEY_OUT_OF_SCOPE_CLASSES.get(coicop, ()):
            out_of_scope[code], _ = _ons_annual_value(
                code,
                source_sheet=sheet,
                consumption_concept="domestic",
                period_value=period_value,
            )
        survey_scope = published - sum(out_of_scope.values())
        frame = float(np.dot(weight, household_draws[column].to_numpy(dtype=float)))
        frame_total += frame
        out_of_scope_total += sum(out_of_scope.values())
        divisions[column] = {
            "coicop": coicop,
            "published_domestic": published,
            "source_record_id": record_id,
            "out_of_scope": out_of_scope,
            "survey_scope": survey_scope,
            "frame": frame,
            "capture": frame / survey_scope if survey_scope > 0 else None,
        }
        if donor_weight is not None:
            donor_values = _numeric(donor[column]).to_numpy(dtype=float)
            donor_total = float(np.dot(donor_weight, donor_values))
            donor_grand_total += donor_total
            divisions[column].update(beside_donor(frame, donor_total))
            divisions[column]["tail"] = division_tail(
                household_draws[column].to_numpy(dtype=float),
                weight,
                donor_values,
                donor_weight,
                frame_sources=sources,
            )
    domestic, _ = _ons_annual_value(
        "0",
        source_sheet="0CN",
        consumption_concept="domestic",
        period_value=period_value,
    )
    national, _ = _ons_annual_value(
        "NAT0",
        source_sheet="0CN",
        consumption_concept="national",
        period_value=period_value,
    )
    total = {
        "frame": frame_total,
        "published_domestic": domestic,
        "published_national": national,
        "out_of_scope": out_of_scope_total,
        "capture_domestic": frame_total / (domestic - out_of_scope_total),
        "capture_national": frame_total / (national - out_of_scope_total),
    }
    if donor_weight is not None:
        total.update(beside_donor(frame_total, donor_grand_total))
    return {
        "resource": UK_LCFS_ONS_EXPENDITURE_RESOURCE,
        "concept": UK_ONS_HFCE_CONCEPT,
        "period_type": "calendar_year",
        "period_value": int(period_value),
        "divisions": divisions,
        "total": total,
        "gated": False,
    }


def division_capture_or_reason(
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    period_value: int,
    donor: pd.DataFrame | None = None,
    sources: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """The division capture, or why it has no figures for this year.

    The block is report-only, so a Consumer Trends row that is missing or
    ambiguous for the year must not stop the stage: the receipt records the
    reason in place of the figures.
    """

    try:
        return division_capture(
            household_draws,
            weights=weights,
            period_value=period_value,
            donor=donor,
            sources=sources,
        )
    except DiagnosticUnavailableError as error:
        return {
            "resource": UK_LCFS_ONS_EXPENDITURE_RESOURCE,
            "concept": UK_ONS_HFCE_CONCEPT,
            "period_type": "calendar_year",
            "period_value": int(period_value),
            "available": False,
            "reason": str(error),
            "gated": False,
        }


def desnz_cars_share_by_fuel(period_value: int) -> dict[str, dict[str, Any]]:
    """The cars share of each fuel's UK road use, from DESNZ's vehicle/fuel ktoe.

    Within one fuel the units cancel, so no ktoe-to-litres factor is needed:
    cars over every vehicle type burning that fuel (petrol: cars, motorcycles
    and LGVs; diesel: cars, buses and coaches, LGVs and HGVs).
    """

    rows = vendored_rows(
        UK_LCFS_ROAD_FUEL_BY_VEHICLE_RESOURCE,
        concept=UK_DESNZ_ROAD_FUEL_CONCEPT,
        period_type="calendar_year",
        period_value=int(period_value),
        geography_id="K02000001",
    )
    shares: dict[str, dict[str, Any]] = {}
    for fuel in UK_LITRES_AUDIT_FUELS.values():
        by_vehicle: dict[str, float] = {}
        record_ids: list[str] = []
        for row in rows:
            dims = row.get("dimensions") or {}
            if dims.get("fuel") != fuel:
                continue
            vehicle = str(dims.get("vehicle_type"))
            if vehicle in by_vehicle:
                raise ValueError(
                    f"{UK_LCFS_ROAD_FUEL_BY_VEHICLE_RESOURCE}: duplicate {vehicle} {fuel} "
                    f"row for {period_value}."
                )
            by_vehicle[vehicle] = float(row["value"])
            record_ids.append(str(row.get("source_record_id", "")))
        total = sum(by_vehicle.values())
        if "cars" not in by_vehicle or total <= 0:
            raise DiagnosticUnavailableError(
                f"{UK_LCFS_ROAD_FUEL_BY_VEHICLE_RESOURCE}: no {fuel} cars row for "
                f"{period_value}."
            )
        shares[fuel] = {
            "ktoe_by_vehicle": dict(sorted(by_vehicle.items())),
            "cars_share": by_vehicle["cars"] / total,
            "source_record_ids": record_ids,
        }
    return shares


def fuel_litres_audit(
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    stage: SourceStageSpec,
) -> dict[str, Any] | None:
    """Frame road-fuel litres against the cars' share of HMRC clearances.

    Diagnostic only (microcosm#890 C7, #1113): for each fuel column the
    declared uprating moves by the vendored litre proxy, the frame's
    prior-weighted spend is divided by the DESNZ pump price of the uprating's
    target year and compared with the HMRC fiscal-year litres of that fuel
    scaled by the cars share of its road use (DESNZ sub-national road fuel by
    vehicle, the uprating's target year; the household frame carries cars, not
    lorries or vans). The OBR cars share of fuel duty receipts, uniform across
    fuels, stays on the receipt beside the total. Recorded in the stage
    evidence; nothing is gated on it.
    """

    parameters = uprating_operation(stage)
    if parameters is None:
        return None
    to_period = int(parameters["to_period"])
    weight_values = np.asarray(weights, dtype=float)
    fuels: dict[str, Any] = {}
    resource = None
    for column, spec in parameters["columns"].items():
        if spec.get("basis") != "vendored_litre_proxy" or column not in household_draws:
            continue
        resource = str(spec["resource"])
        price_rows = vendored_rows(
            resource,
            concept=str(spec["price_concept"]),
            period_type="calendar_year",
            period_value=to_period,
        )
        litre_rows = vendored_rows(
            resource,
            concept=UK_HMRC_LITRES_CONCEPTS[str(column)],
            fiscal_start=f"{to_period}-04-01",
        )
        if len(price_rows) != 1 or len(litre_rows) != 1:
            raise ValueError(
                f"{resource}: litres audit needs one price and one litres row."
            )
        price_pence = float(price_rows[0]["value"])
        hmrc_litres = float(litre_rows[0]["value"])
        spend = _numeric(household_draws[column]).to_numpy(dtype=float)
        spend_total = float(np.dot(spend, weight_values))
        frame_litres = spend_total / (price_pence / 100.0)
        fuels[str(column)] = {
            "price_concept": spec["price_concept"],
            "price_pence_per_litre": price_pence,
            "weighted_spend_gbp": spend_total,
            "frame_litres": frame_litres,
            "hmrc_concept": UK_HMRC_LITRES_CONCEPTS[str(column)],
            "hmrc_litres_all_road_users": hmrc_litres,
        }
    if not fuels:
        return None
    assert resource is not None

    def receipts(category: str) -> float:
        rows = vendored_rows(
            resource,
            concept=UK_OBR_FUEL_DUTY_RECEIPTS_CONCEPT,
            fiscal_start=f"{to_period}-04-01",
            dimensions={"vehicle_category": category},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: litres audit needs one OBR {category!r} row."
            )
        return float(rows[0]["value"])

    cars, total = receipts(UK_OBR_CARS_CATEGORY), receipts(UK_OBR_TOTAL_CATEGORY)
    cars_share = cars / total
    try:
        desnz: dict[str, dict[str, Any]] | None = desnz_cars_share_by_fuel(to_period)
        unavailable = None
    except DiagnosticUnavailableError as error:
        desnz, unavailable = None, str(error)
    for column, entry in fuels.items():
        fuel = UK_LITRES_AUDIT_FUELS[column]
        share = cars_share
        if desnz is not None:
            share = desnz[fuel]["cars_share"]
            entry["desnz_cars_share"] = share
            entry["desnz_ktoe_by_vehicle"] = desnz[fuel]["ktoe_by_vehicle"]
        entry["cars_litres_benchmark"] = entry["hmrc_litres_all_road_users"] * share
        entry["frame_over_cars_benchmark"] = (
            entry["frame_litres"] / entry["cars_litres_benchmark"]
            if entry["cars_litres_benchmark"] > 0
            else None
        )
    frame_total = sum(entry["frame_litres"] for entry in fuels.values())
    benchmark_total = sum(entry["cars_litres_benchmark"] for entry in fuels.values())
    obr_benchmark_total = cars_share * sum(
        entry["hmrc_litres_all_road_users"] for entry in fuels.values()
    )
    return {
        "resource": resource,
        "period_value": to_period,
        "fiscal_start": f"{to_period}-04-01",
        "cars_share_resource": UK_LCFS_ROAD_FUEL_BY_VEHICLE_RESOURCE,
        "obr_fuel_duty_receipts": {
            "cars_gbp": cars,
            "total_gbp": total,
            "cars_share": cars_share,
        },
        "fuels": fuels,
        "frame_litres_total": frame_total,
        "cars_litres_benchmark_total": benchmark_total,
        "frame_over_cars_benchmark": (
            frame_total / benchmark_total if benchmark_total > 0 else None
        ),
        "frame_over_obr_uniform_cars_benchmark": (
            frame_total / obr_benchmark_total if obr_benchmark_total > 0 else None
        ),
        "per_fuel_ratio_basis": (
            UK_LITRES_AUDIT_PER_FUEL_BASIS
            if desnz is not None
            else UK_LITRES_AUDIT_UNIFORM_BASIS
        ),
        **({} if unavailable is None else {"cars_share_unavailable": unavailable}),
        "gated": False,
    }


@dataclass(frozen=True)
class RoadFuelIncidence:
    """Road-fuel draws after the incidence redraw, its receipt and fit records."""

    draws: pd.DataFrame
    receipt: Mapping[str, Any] | None
    fit_weight_records: tuple[FitWeightRecord, ...] = ()


def road_fuel_incidence_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``redraw_zero_road_fuel`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == REDRAW_ZERO_ROAD_FUEL_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def redraw_zero_road_fuel(
    household_draws: pd.DataFrame,
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    flagged: np.ndarray,
    household_ids: np.ndarray,
    weights: Sequence[float],
    parameters: Mapping[str, Any],
) -> RoadFuelIncidence:
    """Give every household the fuel flag marks a positive road-fuel spend.

    The LCFS diary covers two weeks, so about a third of car-owning donors
    record no fuel purchase and the chain reproduces that zero share among the
    households the fuel flag marks as buying petrol or diesel; over a year
    such a car is refuelled. For a flagged household whose chain draw is zero,
    petrol plus diesel is redrawn from the donor's conditional distribution
    among households with positive road fuel, on the chain's predictors: a
    weighted regime-gated QRF draws the total, positive by construction, then
    the petrol share of it (which keeps diesel-only and mixed households), at
    quantiles keyed on the household id. Positive draws are kept; the level
    step then rescales the total (microcosm#1113).
    """

    from microcosm.fit.qrf import Regime, RegimeGatedQRF

    columns = tuple(str(column) for column in parameters.get("columns", ()))
    if columns != UK_LCFS_ROAD_FUEL_COLUMNS:
        raise ValueError(
            f"{REDRAW_ZERO_ROAD_FUEL_KIND} must redraw {UK_LCFS_ROAD_FUEL_COLUMNS}, "
            f"not {columns}."
        )
    if parameters.get("flag") != "has_fuel_consumption":
        raise ValueError(
            f"{REDRAW_ZERO_ROAD_FUEL_KIND} must redraw the has_fuel_consumption "
            "households."
        )
    rule = str(parameters.get("rule") or "")
    if rule != UK_LCFS_ROAD_FUEL_REDRAW_RULE:
        raise ValueError(f"unsupported {REDRAW_ZERO_ROAD_FUEL_KIND} rule {rule!r}.")
    seed = parameters.get("seed")
    n_estimators = parameters.get("n_estimators")
    salt = parameters.get("salt")
    if not isinstance(seed, int) or not isinstance(n_estimators, int):
        raise ValueError(f"{REDRAW_ZERO_ROAD_FUEL_KIND} needs integer seed and trees.")
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"{REDRAW_ZERO_ROAD_FUEL_KIND} needs a declared salt.")
    flagged = np.asarray(flagged, dtype=bool)
    weight = np.asarray(weights, dtype=float)
    spend = household_draws[list(columns)].to_numpy(dtype=float)
    before = spend.sum(axis=1)
    zero = flagged & ~(before > 0)
    redrawn = household_draws.copy()
    regimes: dict[str, str] | None = None
    training_rows = 0
    records: tuple[FitWeightRecord, ...] = ()
    if zero.any():
        donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
            donor, recipient
        )
        donor_total = donor_encoded["petrol_spending"].to_numpy(
            dtype=float
        ) + donor_encoded["diesel_spending"].to_numpy(dtype=float)
        positive = donor_total > 0
        training_rows = int(positive.sum())
        if training_rows == 0:
            raise ValueError("LCFS donor records no positive road-fuel spend.")
        train = donor_encoded.loc[positive, [*predictors, "household_weight"]]
        train = train.reset_index(drop=True)
        train[UK_LCFS_ROAD_FUEL_TOTAL] = donor_total[positive]
        train[UK_LCFS_PETROL_SHARE] = (
            donor_encoded.loc[positive, "petrol_spending"].to_numpy(dtype=float)
            / donor_total[positive]
        )
        targets = [UK_LCFS_ROAD_FUEL_TOTAL, UK_LCFS_PETROL_SHARE]
        fitted = RegimeGatedQRF(n_estimators=n_estimators, seed=seed).fit(
            train, list(predictors), targets, weights="household_weight"
        )
        regimes = {target: str(regime) for target, regime in fitted.regimes().items()}
        if regimes[UK_LCFS_ROAD_FUEL_TOTAL] != Regime.POSITIVE_ONLY:
            raise ValueError(
                f"the road-fuel total fitted regime {regimes[UK_LCFS_ROAD_FUEL_TOTAL]!r}"
                f", not {Regime.POSITIVE_ONLY!r}."
            )
        ids = np.asarray(household_ids)[zero]
        drawn = fitted.predict_from_uniforms(
            recipient_encoded.loc[zero, list(predictors)].reset_index(drop=True),
            quantiles={
                target: stable_identity_uniforms(
                    ids, seed=seed, salt=f"{salt}:{target}"
                )
                for target in targets
            },
            sign_uniforms={
                target: stable_identity_uniforms(
                    ids, seed=seed, salt=f"{salt}:{target}:sign"
                )
                for target in targets
            },
        )
        total = drawn[UK_LCFS_ROAD_FUEL_TOTAL].to_numpy(dtype=float)
        share = np.clip(drawn[UK_LCFS_PETROL_SHARE].to_numpy(dtype=float), 0.0, 1.0)
        rows = np.flatnonzero(zero)
        redrawn.iloc[rows, redrawn.columns.get_loc("petrol_spending")] = share * total
        redrawn.iloc[rows, redrawn.columns.get_loc("diesel_spending")] = (
            1.0 - share
        ) * total
        records = tuple(
            FitWeightRecord(
                f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", fitted.weight_kind
            )
            for target in targets
        )
    after_spend = redrawn[list(columns)].to_numpy(dtype=float)
    after = after_spend.sum(axis=1)

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        mass = float(weight[mask].sum())
        return float(np.dot(weight[mask], values[mask]) / mass) if mass > 0 else None

    flagged_mass = float(weight[flagged].sum())
    redrawn_total = float(np.dot(weight[zero], after[zero]))
    receipt = {
        "operation": REDRAW_ZERO_ROAD_FUEL_KIND,
        "rule": rule,
        "seed": seed,
        "salt": salt,
        "n_estimators": n_estimators,
        "columns": list(columns),
        "regimes": regimes,
        "training_rows": rounded_count(training_rows),
        "flagged_households": int(flagged.sum()),
        "flagged_zero_before": int(zero.sum()),
        "flagged_zero_share_before": (
            float(weight[zero].sum()) / flagged_mass if flagged_mass > 0 else 0.0
        ),
        "flagged_zero_after": int((flagged & ~(after > 0)).sum()),
        "unflagged_with_fuel": int((~flagged & (after > 0)).sum()),
        "mean_positive_before": weighted_mean(before, flagged & (before > 0)),
        "mean_redrawn": weighted_mean(after, zero),
        "mean_flagged_after": weighted_mean(after, flagged),
        "redrawn_petrol_share": (
            float(np.dot(weight[zero], after_spend[zero, 0])) / redrawn_total
            if redrawn_total > 0
            else None
        ),
        "weighted_total_before": float(np.dot(weight, before)),
        "weighted_total_after": float(np.dot(weight, after)),
    }
    return RoadFuelIncidence(redrawn, receipt, records)


def prepayment_meter_from_lcfs(household: pd.DataFrame) -> np.ndarray:
    """1.0 where the LCFS household pays energy through a prepayment meter.

    The rule is ``UK_LCFS_PREPAYMENT_RULE`` (microcosm#1110, the field reading
    of uk-data#539): any key or meter payment positive. Electricity-only
    prepayment counts; a household with no meter payment does not, whether it
    pays by account (b175/b170, b489/b490 net of rebates) or did not answer.
    """

    _require_columns(household, UK_LCFS_PREPAYMENT_CODES)
    prepays = np.zeros(len(household), dtype=bool)
    for code in UK_LCFS_PREPAYMENT_CODES:
        prepays |= _numeric(household[code]).to_numpy(dtype=float) > 0.0
    return prepays.astype(float)


@dataclass(frozen=True)
class PrepaymentMeterDraw:
    """The recipient prepayment flags, their receipt and fit record."""

    flags: np.ndarray
    receipt: Mapping[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def prepayment_meter_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``impute_prepayment_meter`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == IMPUTE_PREPAYMENT_METER_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def impute_prepayment_meter(
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    household_ids: np.ndarray,
    weights: Sequence[float],
    parameters: Mapping[str, Any],
) -> PrepaymentMeterDraw:
    """Draw ``uses_energy_prepayment_meter`` from the LCFS donor (microcosm#1110).

    A weighted regime-gated QRF on the chain's predictors (household counts,
    region, incomes, tenure, accommodation, vehicles) fits the donor's
    prepayment flag; each recipient's draw takes uniforms keyed on its household
    id, so the flag does not depend on row order and no other draw moves. The
    flag is the regime the draw lands in: prepaying or not.
    """

    from microcosm.fit.qrf import RegimeGatedQRF

    if parameters.get("output") != UK_LCFS_PREPAYMENT_METER_COLUMN:
        raise ValueError(
            f"{IMPUTE_PREPAYMENT_METER_KIND} must write "
            f"{UK_LCFS_PREPAYMENT_METER_COLUMN!r}."
        )
    codes = tuple(str(code) for code in parameters.get("codes", ()))
    if codes != UK_LCFS_PREPAYMENT_CODES:
        raise ValueError(
            f"{IMPUTE_PREPAYMENT_METER_KIND} must read {UK_LCFS_PREPAYMENT_CODES}, "
            f"not {codes}."
        )
    rule = str(parameters.get("rule") or "")
    if rule != UK_LCFS_PREPAYMENT_RULE:
        raise ValueError(f"unsupported {IMPUTE_PREPAYMENT_METER_KIND} rule {rule!r}.")
    seed = parameters.get("seed")
    n_estimators = parameters.get("n_estimators")
    salt = parameters.get("salt")
    if not isinstance(seed, int) or not isinstance(n_estimators, int):
        raise ValueError(
            f"{IMPUTE_PREPAYMENT_METER_KIND} needs integer seed and trees."
        )
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"{IMPUTE_PREPAYMENT_METER_KIND} needs a declared salt.")
    donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
        donor, recipient
    )
    target = UK_LCFS_PREPAYMENT_METER_COLUMN
    train = donor_encoded[[*predictors, "household_weight"]].reset_index(drop=True)
    donor_flag = donor[target].to_numpy(dtype=float)
    train[target] = donor_flag
    fitted = RegimeGatedQRF(n_estimators=n_estimators, seed=seed).fit(
        train, list(predictors), [target], weights="household_weight"
    )
    ids = np.asarray(household_ids)
    drawn = fitted.predict_from_uniforms(
        recipient_encoded[list(predictors)].reset_index(drop=True),
        quantiles={target: stable_identity_uniforms(ids, seed=seed, salt=salt)},
        sign_uniforms={
            target: stable_identity_uniforms(ids, seed=seed, salt=f"{salt}:sign")
        },
    )
    flags = drawn[target].to_numpy(dtype=float) > 0.5
    donor_weight = donor["household_weight"].to_numpy(dtype=float)
    weight = np.asarray(weights, dtype=float)
    regions = recipient["region"].astype(str).to_numpy()
    great_britain = regions != "NORTHERN_IRELAND"

    def share(values: np.ndarray, mass: np.ndarray) -> float | None:
        total = float(mass.sum())
        return float(np.dot(mass, values)) / total if total > 0 else None

    receipt = {
        "operation": IMPUTE_PREPAYMENT_METER_KIND,
        "output": target,
        "codes": list(codes),
        "rule": rule,
        "seed": seed,
        "salt": salt,
        "n_estimators": n_estimators,
        "regime": str(fitted.regimes()[target]),
        "donor_share": share(donor_flag, donor_weight),
        "recipient_share": share(flags.astype(float), weight),
        "recipient_share_great_britain": share(
            flags[great_britain].astype(float), weight[great_britain]
        ),
        "recipient_households_great_britain": float(
            weight[great_britain & flags].sum()
        ),
    }
    records = (
        FitWeightRecord(f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", fitted.weight_kind),
    )
    return PrepaymentMeterDraw(flags, receipt, records)


def lcfs_prepayment_meter(
    stage: SourceStageSpec,
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    household_ids: np.ndarray,
    weights: Sequence[float],
) -> PrepaymentMeterDraw | None:
    """Apply the declared ``impute_prepayment_meter`` step (none if undeclared)."""

    parameters = prepayment_meter_operation(stage)
    if parameters is None:
        return None
    return impute_prepayment_meter(
        donor=donor,
        recipient=recipient,
        household_ids=household_ids,
        weights=weights,
        parameters=parameters,
    )


def lcfs_road_fuel_incidence(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    household_ids: np.ndarray,
    weights: Sequence[float],
) -> RoadFuelIncidence:
    """Apply the declared ``redraw_zero_road_fuel`` step (none if undeclared)."""

    parameters = road_fuel_incidence_operation(stage)
    if parameters is None:
        return RoadFuelIncidence(household_draws, None)
    return redraw_zero_road_fuel(
        household_draws,
        donor=donor,
        recipient=recipient,
        flagged=recipient["has_fuel_consumption"].to_numpy(dtype=bool),
        household_ids=household_ids,
        weights=weights,
        parameters=parameters,
    )


@dataclass(frozen=True)
class RoadFuelLevel:
    """The household road-fuel level the ``level_road_fuel`` step scales to.

    ``published`` is the ONS Consumer Trends COICOP 07.2.2 spend (fuels and
    lubricants for personal transport equipment); ``other_fuels_share`` the
    part of it that is neither petrol nor diesel, taken from the LCFS donor's
    own split of 07.2.2. ``level`` is what the frame's petrol plus diesel
    totals at prior weights after the step.
    """

    published: float
    other_fuels_share: float
    receipt: Mapping[str, Any]

    @property
    def level(self) -> float:
        return self.published * (1.0 - self.other_fuels_share)


def road_fuel_level_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``level_road_fuel`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == LEVEL_ROAD_FUEL_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def lcfs_other_road_fuel_share(lcfs_household: pd.DataFrame) -> float:
    """The donor's weighted share of COICOP 07.2.2 that is neither petrol nor diesel."""

    household = _lowercase(lcfs_household)
    _require_columns(
        household,
        (
            UK_LCFS_DONOR_WEIGHT_COLUMN,
            "c72211",
            "c72212",
            *UK_LCFS_OTHER_ROAD_FUEL_CODES,
        ),
    )
    weight = _numeric(household[UK_LCFS_DONOR_WEIGHT_COLUMN]).to_numpy(dtype=float)

    def spend(code: str) -> np.ndarray:
        # Floored at zero, as the imputation's donor is (microcosm#1063 c9).
        return np.maximum(_numeric(household[code]).to_numpy(dtype=float), 0.0)

    other = sum(spend(code) for code in UK_LCFS_OTHER_ROAD_FUEL_CODES)
    total = spend("c72211") + spend("c72212") + other
    denominator = float(np.dot(weight, total))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("LCFS donor records no COICOP 07.2.2 road-fuel spend.")
    return float(np.dot(weight, other)) / denominator


def road_fuel_level(
    parameters: Mapping[str, Any], *, other_fuels_share: float
) -> RoadFuelLevel:
    """Resolve the declared ONS road-fuel level from the vendored rows."""

    columns = tuple(str(column) for column in parameters.get("columns", ()))
    if columns != UK_LCFS_ROAD_FUEL_COLUMNS:
        raise ValueError(
            f"{LEVEL_ROAD_FUEL_KIND} must level {UK_LCFS_ROAD_FUEL_COLUMNS}, "
            f"not {columns}."
        )
    resource = str(parameters.get("resource") or "")
    if resource != UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE:
        raise ValueError(
            f"{LEVEL_ROAD_FUEL_KIND} must read {UK_LCFS_ROAD_FUEL_LEVEL_RESOURCE!r}, "
            f"not {resource!r}."
        )
    rule = str(parameters.get("other_fuels_rule") or "")
    if rule != UK_LCFS_ROAD_FUEL_OTHER_SHARE_RULE:
        raise ValueError(
            f"unsupported {LEVEL_ROAD_FUEL_KIND} other_fuels_rule {rule!r}."
        )
    codes = tuple(str(code) for code in parameters.get("other_fuels_codes", ()))
    if codes != UK_LCFS_OTHER_ROAD_FUEL_CODES:
        raise ValueError(
            f"{LEVEL_ROAD_FUEL_KIND} other_fuels_codes must be "
            f"{UK_LCFS_OTHER_ROAD_FUEL_CODES}, not {codes}."
        )
    if str(parameters.get("period_type")) != "calendar_year":
        raise ValueError(f"{LEVEL_ROAD_FUEL_KIND} binds a calendar-year ONS level.")
    if not 0.0 <= other_fuels_share < 1.0:
        raise ValueError(f"other-fuels share {other_fuels_share} is outside [0, 1).")
    period_value = int(parameters["period_value"])
    rows = vendored_rows(
        resource,
        concept=str(parameters["concept"]),
        period_type="calendar_year",
        period_value=period_value,
        geography_id="K02000001",
        dimensions={"coicop": str(parameters["coicop"]), "frequency": "annual"},
    )
    if len(rows) != 1:
        raise ValueError(
            f"{resource}: expected one {parameters['coicop']} row for "
            f"{period_value}, found {len(rows)}."
        )
    published = float(rows[0]["value"])
    if not np.isfinite(published) or published <= 0:
        raise ValueError(f"{resource}: the road-fuel level must be positive.")
    receipt = {
        "operation": LEVEL_ROAD_FUEL_KIND,
        "resource": resource,
        "concept": str(parameters["concept"]),
        "coicop": str(parameters["coicop"]),
        "period_type": "calendar_year",
        "period_value": period_value,
        "source_record_id": str(rows[0].get("source_record_id", "")),
        "published": published,
        "other_fuels_rule": rule,
        "other_fuels_codes": list(UK_LCFS_OTHER_ROAD_FUEL_CODES),
        "other_fuels_share": float(other_fuels_share),
        "level": published * (1.0 - other_fuels_share),
        "columns": list(UK_LCFS_ROAD_FUEL_COLUMNS),
    }
    return RoadFuelLevel(published, float(other_fuels_share), receipt)


def level_road_fuel(
    household_draws: pd.DataFrame,
    *,
    level: RoadFuelLevel,
    weights: Sequence[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale petrol and diesel by one factor so the prior-weighted total is the level.

    One factor keeps the drawn petrol/diesel mix and every household's
    relative spend; only the level moves (microcosm#1113: the LCFS diary
    records about a quarter less road fuel than ONS household spending).
    """

    columns = list(UK_LCFS_ROAD_FUEL_COLUMNS)
    weight_values = np.asarray(weights, dtype=float)
    spend = household_draws[columns].to_numpy(dtype=float)
    scaled, before, factor = _scale_to_level(
        spend, weight_values, level.level, what="road-fuel spend"
    )
    result = household_draws.copy()
    result[columns] = scaled
    after = result[columns].to_numpy(dtype=float)
    petrol = float(np.dot(weight_values, after[:, 0]))
    receipt = {
        **dict(level.receipt),
        "frame_before": before,
        "factor": float(factor),
        "frame_after": float(np.dot(weight_values, after.sum(axis=1))),
        "petrol_share_of_level": petrol / level.level,
    }
    return result, receipt


def lcfs_road_fuel_level(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    lcfs_household: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply the declared ``level_road_fuel`` step (none if undeclared)."""

    parameters = road_fuel_level_operation(stage)
    if parameters is None:
        return household_draws, None
    level = road_fuel_level(
        parameters, other_fuels_share=lcfs_other_road_fuel_share(lcfs_household)
    )
    return level_road_fuel(household_draws, level=level, weights=weights)


def _scale_to_level(
    values: np.ndarray, weights: np.ndarray, level: float, *, what: str
) -> tuple[np.ndarray, float, float]:
    """Scale ``values`` by the one factor that takes their weighted total to ``level``.

    ``values`` is one column, or several scaled together by the same factor
    so their mix is kept. Returns the scaled values, the weighted total before
    and the factor; a frame with nothing to scale is refused.
    """

    before = float(np.dot(weights, values if values.ndim == 1 else values.sum(axis=1)))
    if not np.isfinite(before) or before <= 0:
        raise ValueError(f"the frame draws no {what} to level.")
    factor = level / before
    return values * factor, before, factor


@dataclass(frozen=True)
class ExciseQuantityLevel:
    """One engine input's national quantity a year, from its published rows.

    ``annual`` is in the engine's units a year: litres of the product, or
    cigarettes, or grams, or pounds; the step divides the weekly tobacco
    inputs by ``UK_PE_UK_WEEKS_IN_YEAR`` after levelling. ``key`` is the
    household spending the quantity is shared out by: alcohol, tobacco, the
    National Lottery's stakes or the betting and gaming stakes.
    """

    column: str
    annual: float
    key: str
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class AlcoholVolumeWeights:
    """How off- and on-trade alcohol spending share the national volume."""

    rule: str
    on_trade_share: float | None
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class ExciseLevels:
    """Everything the step levels to: quantities, draught shares, the volume rule."""

    quantities: Mapping[str, ExciseQuantityLevel]
    draught: Mapping[str, Mapping[str, Any]]
    volume: AlcoholVolumeWeights


def excise_quantities_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``level_excise_quantities`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == LEVEL_EXCISE_QUANTITIES_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def _one_vendored_value(
    resource: str,
    parameters: Mapping[str, Any],
    declared: Mapping[str, Any],
    *,
    what: str,
    unit: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """The one published value a declaration names, for the step's fiscal year.

    With ``unit``, the row must be stored in it: a value read in another unit
    would be off by the publisher's scale. A declaration may name its own
    ``geography_id`` in place of the step's.
    """

    concept = str(declared.get("concept") or "")
    dimensions = declared.get("dimensions") or {}
    if not concept or not isinstance(dimensions, Mapping):
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: {what} must name a concept and "
            "its dimensions."
        )
    geography = str(declared.get("geography_id") or parameters["geography_id"])
    rows = vendored_rows(
        resource,
        fiscal_start=str(parameters["fiscal_start"]),
        concept=concept,
        period_type="fiscal_year",
        period_value=int(parameters["period_value"]),
        geography_id=geography,
        dimensions=dict(dimensions),
    )
    if len(rows) != 1:
        raise ValueError(
            f"{resource}: expected one {concept} row {dict(dimensions)} for fiscal "
            f"{parameters['period_value']} in {geography}, found {len(rows)}."
        )
    if unit is not None and rows[0].get("unit") != unit:
        raise ValueError(
            f"{resource}: {what}'s row is stored in {rows[0].get('unit')!r}, not the "
            f"declared {unit!r}."
        )
    value = float(rows[0]["value"])
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{resource}: {what}'s published value must be nonnegative.")
    receipt = {
        "concept": concept,
        "dimensions": dict(dimensions),
        "geography_id": geography,
        "source_record_id": str(rows[0].get("source_record_id", "")),
        "published": value,
    }
    return value, receipt


def _declared_rate(declared: object, *, what: str) -> tuple[float, dict[str, Any]]:
    """A declared divisor: its value in (0, 1] and the engine parameter it mirrors."""

    if not isinstance(declared, Mapping) or not declared.get("parameter_path"):
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: {what} must give its value and the "
            "engine parameter it mirrors."
        )
    value = float(declared.get("value"))
    if not 0.0 < value <= 1.0:
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: {what} {value} is outside (0, 1]."
        )
    return value, dict(declared)


def excise_levels(parameters: Mapping[str, Any]) -> ExciseLevels:
    """Resolve everything the declared step levels to from the vendored rows.

    ``outputs`` are the alcohol and tobacco inputs. Each names its published
    row (concept and dimensions, for the declared fiscal year and geography),
    the unit the row is stored in, the share of the row it takes and, for
    litres of pure alcohol, the strength that turns them into litres of the
    product: the declared value, with the engine parameter it mirrors, which
    policyengine-uk multiplies back. Outputs that share a row must take all
    of it between them. ``draught_shares`` are published ratios the step
    writes as they are; ``gambling`` declares the series and rates the
    gambling bases come from (``gambling_levels``); ``alcohol_volume_weights``
    declares how off- and on-trade spending share the alcohol volume.
    """

    resource = str(parameters.get("resource") or "")
    if resource != UK_LCFS_EXCISE_FACTS_RESOURCE:
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND} must read "
            f"{UK_LCFS_EXCISE_FACTS_RESOURCE!r}, not {resource!r}."
        )
    if str(parameters.get("period_type")) != "fiscal_year":
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND} levels to fiscal-year clearances, the "
            "year policyengine-uk charges excise over."
        )
    if parameters.get("weeks_per_year") != UK_PE_UK_WEEKS_IN_YEAR:
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND} must declare weeks_per_year: "
            f"{UK_PE_UK_WEEKS_IN_YEAR}, policyengine-uk's own."
        )
    for key in ("period_value", "fiscal_start", "geography_id"):
        if parameters.get(key) in (None, ""):
            raise ValueError(f"{LEVEL_EXCISE_QUANTITIES_KIND} must declare {key}.")
    outputs = parameters.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ValueError(f"{LEVEL_EXCISE_QUANTITIES_KIND} declares no outputs.")
    levels: dict[str, ExciseQuantityLevel] = {}
    taken: dict[str, float] = {}
    for column, declared in outputs.items():
        if column not in UK_LCFS_EXCISE_OUTPUT_UNITS:
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND} cannot write {column!r} as a "
                "quantity; its quantities are the policyengine-uk inputs "
                f"{tuple(UK_LCFS_EXCISE_OUTPUT_UNITS)}."
            )
        if not isinstance(declared, Mapping):
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: {column} is not a mapping."
            )
        unit = str(declared.get("unit") or "")
        if unit not in UK_LCFS_EXCISE_OUTPUT_UNITS[column]:
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: {column} is declared in {unit!r}, "
                f"not one of {UK_LCFS_EXCISE_OUTPUT_UNITS[column]}."
            )
        share = float(declared.get("share", 1.0))
        if not 0.0 < share <= 1.0:
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: {column} takes a share {share} "
                "outside (0, 1]."
            )
        strength = declared.get("strength")
        if (unit == UK_LCFS_EXCISE_ALCOHOL_UNIT) != (strength is not None):
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: {column} needs a strength exactly "
                "when it is declared in litres of pure alcohol."
            )
        divisor = 1.0
        if strength is not None:
            divisor, _ = _declared_rate(strength, what=f"{column}'s strength")
        published, receipt = _one_vendored_value(
            resource, parameters, declared, what=column, unit=unit
        )
        if published <= 0:
            raise ValueError(
                f"{resource}: {column}'s published quantity must be positive."
            )
        row = receipt["source_record_id"]
        taken[row] = taken.get(row, 0.0) + share
        annual = published * UK_LCFS_EXCISE_UNITS[unit] * share / divisor
        levels[column] = ExciseQuantityLevel(
            column,
            annual,
            "alcohol" if column in UK_LCFS_EXCISE_ALCOHOL_OUTPUTS else "tobacco",
            {
                **receipt,
                "unit": unit,
                "share": share,
                "strength": None if strength is None else dict(strength),
                "annual_level": annual,
            },
        )
    partial = {row: total for row, total in taken.items() if abs(total - 1.0) > 1e-9}
    if partial:
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: the outputs take {partial} of their "
            "published rows, not all of each."
        )
    if "gambling" in parameters:
        levels.update(gambling_levels(resource, parameters))
    return ExciseLevels(
        quantities=levels,
        draught=draught_shares(resource, parameters),
        volume=alcohol_volume_weights(resource, parameters),
    )


def _summed_rows(
    resource: str,
    parameters: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    side: str,
    what: str,
    unit: str,
) -> tuple[float, list[dict[str, Any]]]:
    """The sum of the published rows a ratio's ``side`` names, and their receipts."""

    cells = entry.get(side)
    if not isinstance(cells, list | tuple) or not cells:
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: {what} declares no {side} rows."
        )
    read = [
        _one_vendored_value(
            resource,
            parameters,
            {"concept": entry.get("concept"), "dimensions": cell},
            what=f"{what} {side}",
            unit=unit,
        )
        for cell in cells
    ]
    return sum(value for value, _ in read), [receipt for _, receipt in read]


def draught_shares(
    resource: str, parameters: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Each declared draught share: its published numerator rows over its total.

    HMRC splits beer and cider clearances by the relief they took; the
    share is the clearances under draught relief, alone or with small
    producer relief, over all of that product's clearances, in litres of
    pure alcohol. policyengine-uk applies it to the household's litres, so
    the strength the litres were divided by cancels out of it.
    """

    declared = parameters.get("draught_shares") or {}
    if not isinstance(declared, Mapping):
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: draught_shares is not a mapping."
        )
    shares: dict[str, dict[str, Any]] = {}
    for column, entry in declared.items():
        if column not in UK_LCFS_EXCISE_DRAUGHT_OUTPUTS:
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND} cannot write {column!r} as a draught "
                f"share; it writes {UK_LCFS_EXCISE_DRAUGHT_OUTPUTS}."
            )
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: {column} is not a mapping."
            )
        unit = str(entry.get("unit") or "")
        if unit != UK_LCFS_EXCISE_ALCOHOL_UNIT:
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: {column} must be declared in "
                f"{UK_LCFS_EXCISE_ALCOHOL_UNIT!r}, not {unit!r}."
            )

        numerator, numerator_rows = _summed_rows(
            resource, parameters, entry, side="numerator", what=column, unit=unit
        )
        denominator, denominator_rows = _summed_rows(
            resource, parameters, entry, side="denominator", what=column, unit=unit
        )
        if denominator <= 0 or not 0.0 <= numerator <= denominator:
            raise ValueError(
                f"{resource}: {column}'s published rows do not give a share in [0, 1]."
            )
        shares[column] = {
            "share": numerator / denominator,
            "numerator": numerator_rows,
            "denominator": denominator_rows,
        }
    return shares


def gambling_levels(
    resource: str, parameters: Mapping[str, Any]
) -> dict[str, ExciseQuantityLevel]:
    """Each gambling input's national base a year, in pounds (microcosm#1359).

    The declaration names the published series (HMRC's betting and gaming
    receipts by duty, Machine Games Duty liabilities by rate band and the
    Gambling Commission's gross gambling yield, with its own geography) and
    the rates, each the base year's value of the engine parameter it names.
    ``UK_LCFS_GAMBLING_BASES`` turns a series into each single-duty input's
    base; General Betting Duty's base is split over its inputs by the yield
    cells of ``UK_LCFS_GENERAL_BETTING_YIELD_CELLS``. Charged at the engine's
    rates, the bases give back the published receipts and liabilities, as
    the clearances give back alcohol and tobacco duty.
    """

    declared = parameters.get("gambling")
    if not isinstance(declared, Mapping) or set(declared) != {
        "receipts",
        "machine_games_liabilities",
        "yield",
        "rates",
    }:
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: gambling must declare receipts, "
            "machine_games_liabilities, yield and rates."
        )
    rates_declared = declared["rates"]
    if not isinstance(rates_declared, Mapping) or set(rates_declared) != set(
        UK_LCFS_GAMBLING_RATES
    ):
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND}: gambling rates must be "
            f"{UK_LCFS_GAMBLING_RATES}."
        )
    rates = {
        name: _declared_rate(rates_declared[name], what=f"the {name} rate")
        for name in UK_LCFS_GAMBLING_RATES
    }
    for name in ("receipts", "machine_games_liabilities", "yield"):
        if not isinstance(declared[name], Mapping):
            raise ValueError(
                f"{LEVEL_EXCISE_QUANTITIES_KIND}: gambling {name} is not a mapping."
            )
    dimension = {"receipts": "gambling_duty", "machine_games_liabilities": "mgd_rate"}

    def cell(name: str, dimensions: Mapping[str, str]) -> dict[str, Any]:
        # A series may pin dimensions of its own, such as the accounting basis.
        source = declared[name]
        pinned = dict(source.get("dimensions") or {})
        return {**dict(source), "dimensions": {**pinned, **dimensions}}

    def series(name: str, value: str) -> tuple[float, dict[str, Any]]:
        return _one_vendored_value(
            resource,
            parameters,
            cell(name, {dimension[name]: value}),
            what=f"{name} {value}",
            unit=UK_LCFS_EXCISE_GAMBLING_UNIT,
        )

    levels: dict[str, ExciseQuantityLevel] = {}
    for column, (source, value, rate_name) in UK_LCFS_GAMBLING_BASES.items():
        published, receipt = series(source, value)
        rate, rate_receipt = rates[rate_name]
        annual = published / rate
        levels[column] = ExciseQuantityLevel(
            column,
            annual,
            UK_LCFS_NATIONAL_LOTTERY_KEY
            if column == UK_LCFS_NATIONAL_LOTTERY_INPUT
            else UK_LCFS_BETTING_AND_GAMING_KEY,
            {
                **receipt,
                "unit": UK_LCFS_EXCISE_GAMBLING_UNIT,
                "rate": rate_receipt,
                "share": 1.0,
                "annual_level": annual,
            },
        )
    receipts, receipt = series("receipts", "general_betting_duty")
    rate, rate_receipt = rates["general_betting"]
    base = receipts / rate
    cells: dict[str, tuple[float, dict[str, Any]]] = {}
    for name, (sector, activity, event) in UK_LCFS_GENERAL_BETTING_YIELD_CELLS.items():
        cells[name] = _one_vendored_value(
            resource,
            parameters,
            cell(
                "yield",
                {
                    "gambling_sector": sector,
                    "gambling_activity": activity,
                    "betting_event": event,
                },
            ),
            what=f"the {name} yield",
            unit=UK_LCFS_EXCISE_GAMBLING_UNIT,
        )
    value = {name: cell[0] for name, cell in cells.items()}
    if value["remote_all_events"] <= 0:
        raise ValueError(f"{resource}: remote betting has no published yield.")
    horses = value["remote_horses"] / value["remote_all_events"]
    parts = {
        "land_based_fixed_odds_betting_net_losses": value["over_the_counter"],
        "remote_uk_horseracing_betting_net_losses": value["remote_fixed_odds"] * horses,
        "remote_other_betting_net_losses": value["remote_fixed_odds"] * (1.0 - horses),
        "betting_exchange_commission": value["exchange"],
        "horse_and_dog_pool_betting_net_losses": value["pools_non_remote"]
        + value["pools_remote"],
    }
    whole = sum(parts.values())
    if whole <= 0:
        raise ValueError(f"{resource}: betting has no published yield to split.")
    yield_rows = {name: cell[1] for name, cell in cells.items()}
    for column in UK_LCFS_GENERAL_BETTING_OUTPUTS:
        share = parts[column] / whole
        annual = base * share
        levels[column] = ExciseQuantityLevel(
            column,
            annual,
            UK_LCFS_BETTING_AND_GAMING_KEY,
            {
                **receipt,
                "unit": UK_LCFS_EXCISE_GAMBLING_UNIT,
                "rate": rate_receipt,
                "share": share,
                "split_by": yield_rows,
                "remote_horse_racing_share": horses,
                "annual_level": annual,
            },
        )
    return levels


def alcohol_volume_weights(
    resource: str, parameters: Mapping[str, Any]
) -> AlcoholVolumeWeights:
    """The declared rule that shares alcohol volume between the two channels."""

    declared = parameters.get("alcohol_volume_weights")
    if not isinstance(declared, Mapping):
        raise ValueError(
            f"{LEVEL_EXCISE_QUANTITIES_KIND} must declare alcohol_volume_weights."
        )
    rule = str(declared.get("rule") or "")
    if rule == UK_LCFS_ALCOHOL_VOLUME_BY_SPENDING:
        return AlcoholVolumeWeights(rule, None, {"rule": rule})
    if rule != UK_LCFS_ALCOHOL_VOLUME_BY_CHANNEL:
        raise ValueError(
            f"unsupported {LEVEL_EXCISE_QUANTITIES_KIND} alcohol_volume_weights "
            f"rule {rule!r}."
        )
    share, receipt = _one_vendored_value(
        resource, parameters, declared, what="the on-trade share"
    )
    if not 0.0 < share < 1.0:
        raise ValueError(f"{resource}: the on-trade share {share} is outside (0, 1).")
    return AlcoholVolumeWeights(rule, share, {"rule": rule, **receipt})


def level_excise_quantities(
    household_draws: pd.DataFrame,
    *,
    levels: ExciseLevels,
    weights: Sequence[float],
    participation: Mapping[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Give each household its part of each national quantity, by its spending.

    Each household's alcohol inputs are the national volumes in proportion to
    its alcohol spending, its tobacco inputs the national clearances in
    proportion to its tobacco spending, its National Lottery ticket spending
    the national base in proportion to its National Lottery stakes and its
    other gambling inputs the national bases in proportion to its betting and
    gaming stakes, all at prior weights: one factor per input, the
    ``level_road_fuel`` scaling, so the frame's weighted total is the
    published quantity. Under the published on-trade share each channel's
    volume goes to that channel's spenders; by spending, a pound buys the same
    volume in either, which leans the alcohol inputs towards drinkers in pubs
    (``UK_LCFS_ALCOHOL_VOLUME_BY_SPENDING`` records by how much). The betting
    and gaming bases all follow the same stakes, since the diary cannot tell
    remote from land-based play. Weekly inputs are the annual quantity over
    the engine's 52 weeks; a draught share is written as published. The
    spending is the drawn shares of the drawn totals, so no total moves
    (microcosm#1359). ``participation`` gives the two gambling keys the
    declared ``top_up_gambling_participation`` step brought to published
    participation, in place of the drawn stakes; the stored spending stays
    the drawn one.
    """

    weight_values = np.asarray(weights, dtype=float)
    mass = float(weight_values.sum())
    spending = excise_spending(household_draws)
    off_trade = spending[UK_LCFS_OFF_TRADE_ALCOHOL_SPENDING].to_numpy(dtype=float)
    on_trade = spending[UK_LCFS_ON_TRADE_ALCOHOL_SPENDING].to_numpy(dtype=float)
    tobacco = spending[UK_LCFS_TOBACCO_SPENDING].to_numpy(dtype=float)
    stakes = spending[UK_LCFS_GAMBLING_SPENDING].to_numpy(dtype=float)
    lottery = spending[UK_LCFS_NATIONAL_LOTTERY_SPENDING].to_numpy(dtype=float)
    other_stakes = spending[UK_LCFS_OTHER_GAMBLING_SPENDING].to_numpy(dtype=float)
    betting = spending[UK_LCFS_BETTING_AND_GAMING_SPENDING].to_numpy(dtype=float)
    volume = levels.volume
    if volume.rule == UK_LCFS_ALCOHOL_VOLUME_BY_SPENDING:
        alcohol = off_trade + on_trade
    else:
        totals = (float(weight_values @ off_trade), float(weight_values @ on_trade))
        if min(totals) <= 0:
            raise ValueError("the frame draws no alcohol in one of the two channels.")
        share = float(volume.on_trade_share)
        alcohol = (1.0 - share) * off_trade / totals[0] + share * on_trade / totals[1]
    keys = {
        "alcohol": alcohol,
        "tobacco": tobacco,
        UK_LCFS_NATIONAL_LOTTERY_KEY: lottery,
        UK_LCFS_BETTING_AND_GAMING_KEY: betting,
    }
    if participation is not None:
        if set(participation) != set(UK_GAMBLING_PARTICIPATION_KEYS):
            raise ValueError(
                f"the topped-up gambling keys must be {UK_GAMBLING_PARTICIPATION_KEYS}, "
                f"not {sorted(participation)}."
            )
        for key, values in participation.items():
            topped = np.asarray(values, dtype=float)
            if topped.shape != (len(household_draws),):
                raise ValueError(f"the topped-up {key} key is not one per household.")
            keys[key] = topped
    quantities = pd.DataFrame(index=household_draws.index)
    outputs: dict[str, Any] = {}
    for column, level in levels.quantities.items():
        annual, before, factor = _scale_to_level(
            keys[level.key], weight_values, level.annual, what=f"{level.key} spending"
        )
        weekly = column in UK_LCFS_EXCISE_TOBACCO_OUTPUTS
        quantities[column] = annual / UK_PE_UK_WEEKS_IN_YEAR if weekly else annual
        stored = quantities[column].to_numpy(dtype=float)
        outputs[column] = {
            **dict(level.receipt),
            "key": level.key,
            "key_total": before,
            "factor": float(factor),
            "stored_per": "week" if weekly else "year",
            "frame_annual_total": float(weight_values @ stored)
            * (UK_PE_UK_WEEKS_IN_YEAR if weekly else 1.0),
            "positive_share": float(weight_values[stored > 0].sum() / mass),
        }
    for column, draught in levels.draught.items():
        quantities[column] = float(draught["share"])
    components = {
        name: {
            "weighted_total": float(weight_values @ values),
            "positive_share": float(weight_values[values > 0].sum() / mass),
        }
        for name, values in (
            (UK_LCFS_TOBACCO_SPENDING, tobacco),
            (UK_LCFS_OFF_TRADE_ALCOHOL_SPENDING, off_trade),
            (UK_LCFS_ON_TRADE_ALCOHOL_SPENDING, on_trade),
            (UK_LCFS_GAMBLING_SPENDING, stakes),
            (UK_LCFS_NATIONAL_LOTTERY_SPENDING, lottery),
            (UK_LCFS_OTHER_GAMBLING_SPENDING, other_stakes),
            (UK_LCFS_BETTING_AND_GAMING_SPENDING, betting),
        )
    }
    receipt = {
        "operation": LEVEL_EXCISE_QUANTITIES_KIND,
        "keys": {
            key: {"positive_share": float(weight_values[values > 0].sum() / mass)}
            for key, values in keys.items()
        },
        "components": components,
        "alcohol_volume_weights": dict(volume.receipt),
        "outputs": outputs,
        "draught_shares": {
            column: dict(draught) for column, draught in levels.draught.items()
        },
    }
    return quantities, receipt


def lcfs_excise_quantities(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    weights: Sequence[float],
    shares: Mapping[str, object],
    participation: Mapping[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Apply the declared ``level_excise_quantities`` step (none if undeclared)."""

    parameters = excise_quantities_operation(stage)
    if parameters is None:
        return None, None
    if set(shares) != set(UK_LCFS_EXCISE_SHARES):
        raise ValueError(
            f"{stage.stage}: {LEVEL_EXCISE_QUANTITIES_KIND} needs the derive step's "
            f"excise shares {tuple(UK_LCFS_EXCISE_SHARES)}."
        )
    levels = excise_levels(parameters)
    quantities, receipt = level_excise_quantities(
        household_draws, levels=levels, weights=weights, participation=participation
    )
    receipt = {
        "operation": LEVEL_EXCISE_QUANTITIES_KIND,
        "resource": str(parameters["resource"]),
        "period_type": "fiscal_year",
        "period_value": int(parameters["period_value"]),
        "fiscal_start": str(parameters["fiscal_start"]),
        "geography_id": str(parameters["geography_id"]),
        "weeks_per_year": UK_PE_UK_WEEKS_IN_YEAR,
        **receipt,
    }
    return quantities, receipt


#: The gambling top-up's recreation feature: recreation less its gambling
#: stakes, which P609 includes.
UK_LCFS_RECREATION_NET_OF_STAKES = "recreation_net_of_gambling_stakes"


def gambling_participation_features(
    donor_encoded: pd.DataFrame,
    donor_spending: pd.DataFrame,
    recipient_encoded: pd.DataFrame,
    drawn: pd.DataFrame,
    *,
    predictors: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The donor's and the frame's features for the gambling top-up's two fits.

    The chain's predictors and the spend targets the shares were drawn on:
    the donor's diary totals, and the chain's raw draws on the frame. Recreation
    enters net of its gambling stakes on both sides. P609 sums the stake lines,
    so a recorded stake would otherwise raise the very feature that predicts it,
    and the walk would favour high-recreation households for that reason alone
    (Vahid's review of microcosm#1381, round 4).
    """

    spend = [
        c for c in UK_LCFS_CONSUMPTION_TARGET_COLUMNS if c != "recreation_consumption"
    ]
    donor_features = donor_encoded[[*predictors, *spend]].reset_index(drop=True)
    donor_features[UK_LCFS_RECREATION_NET_OF_STAKES] = donor_encoded[
        "recreation_consumption"
    ].to_numpy(dtype=float) - donor_spending[UK_LCFS_GAMBLING_SPENDING].to_numpy(
        dtype=float
    )
    recipient_features = recipient_encoded[list(predictors)].reset_index(drop=True)
    for column in spend:
        recipient_features[column] = drawn[column].to_numpy(dtype=float)
    share = np.clip(drawn[UK_LCFS_GAMBLING_SHARE].to_numpy(dtype=float), 0.0, 1.0)
    recipient_features[UK_LCFS_RECREATION_NET_OF_STAKES] = drawn[
        "recreation_consumption"
    ].to_numpy(dtype=float) * (1.0 - share)
    return donor_features, recipient_features


def lcfs_gambling_participation(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    drawn: pd.DataFrame,
    person: pd.DataFrame,
    household_ids: np.ndarray,
    weights: Sequence[float],
    sources: np.ndarray | None,
    shares: Mapping[str, object],
) -> GamblingParticipation | None:
    """Apply the declared ``top_up_gambling_participation`` step (none if undeclared).

    The keys are the drawn National Lottery and betting and gaming stakes,
    and the donor's own. Both fits read ``gambling_participation_features``.
    The step only re-keys ``level_excise_quantities``, so it needs that step
    and the shares.
    """

    parameters = gambling_participation_operation(stage)
    if parameters is None:
        return None
    if excise_quantities_operation(stage) is None or set(shares) != set(
        UK_LCFS_EXCISE_SHARES
    ):
        raise ValueError(
            f"{stage.stage}: {TOP_UP_GAMBLING_PARTICIPATION_KIND} re-keys "
            f"{LEVEL_EXCISE_QUANTITIES_KIND}, so it needs that step and the "
            f"excise shares {tuple(UK_LCFS_EXCISE_SHARES)}."
        )
    columns = {
        UK_LCFS_NATIONAL_LOTTERY_KEY: UK_LCFS_NATIONAL_LOTTERY_SPENDING,
        UK_LCFS_BETTING_AND_GAMING_KEY: UK_LCFS_BETTING_AND_GAMING_SPENDING,
    }
    spending = excise_spending(household_draws)
    donor_spending = excise_spending(donor)
    donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
        donor, recipient
    )
    donor_features, recipient_features = gambling_participation_features(
        donor_encoded,
        donor_spending,
        recipient_encoded,
        drawn,
        predictors=predictors,
    )
    return top_up_gambling_participation(
        {
            key: spending[column].to_numpy(dtype=float)
            for key, column in columns.items()
        },
        person=person,
        household_ids=household_ids,
        weights=weights,
        sources=sources,
        donor_features=donor_features,
        donor_keys={
            key: donor_spending[column].to_numpy(dtype=float)
            for key, column in columns.items()
        },
        donor_weights=donor_encoded["household_weight"].to_numpy(dtype=float),
        recipient_features=recipient_features,
        parameters=parameters,
        fit_name=UK_LCFS_CONSUMPTION_FIT_NAME,
    )


def recompose_from_remainder_operation(
    stage: SourceStageSpec,
) -> Mapping[str, Any] | None:
    """The stage's declared ``recompose_from_remainder`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == RECOMPOSE_FROM_REMAINDER_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def check_recomposed_parents(parameters: Mapping[str, Any]) -> None:
    """Refuse a declaration that differs from the totals the stage recomposes."""

    parents = parameters.get("parents")
    if not isinstance(parents, Mapping) or set(parents) != set(
        UK_LCFS_RECOMPOSED_PARENTS
    ):
        raise ValueError(
            f"{RECOMPOSE_FROM_REMAINDER_KIND} must declare "
            f"{sorted(UK_LCFS_RECOMPOSED_PARENTS)}."
        )
    for parent, (subtracted, components) in UK_LCFS_RECOMPOSED_PARENTS.items():
        declared = parents[parent]
        if (
            not isinstance(declared, Mapping)
            or tuple(declared.get("drawn_subtracts", ())) != subtracted
            or tuple(declared.get("components", ())) != components
        ):
            raise ValueError(
                f"{RECOMPOSE_FROM_REMAINDER_KIND} declares {parent!r} as "
                f"{declared!r}, not its draw less {subtracted} plus {components}."
            )
    if parameters.get("maximum_total") != UK_LCFS_RECOMPOSED_MAXIMUM_TOTAL:
        raise ValueError(
            f"{RECOMPOSE_FROM_REMAINDER_KIND} must declare maximum_total: "
            f"{UK_LCFS_RECOMPOSED_MAXIMUM_TOTAL}."
        )
    levelling = parameters.get("level_division_excess")
    if levelling is not None and dict(levelling) != {
        "rule": UK_LCFS_LEVEL_DIVISION_EXCESS,
        "basis": UK_LCFS_LEVEL_DIVISION_BASIS,
        "divisions": list(UK_LCFS_COICOP_DIVISION_COLUMNS),
    }:
        raise ValueError(
            f"{RECOMPOSE_FROM_REMAINDER_KIND} declares level_division_excess "
            f"{levelling!r}, not the rule the stage applies."
        )


def published_fuel_spend(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The published spend on the heating fuels the stage draws.

    ONS Consumer Trends classes read from the vendored rows the declaration
    names: the liquid and solid fuels of COICOP 04.5, each beside the column
    that carries it (``UK_LCFS_FUEL_PARTS``). Report-only: the stage draws the
    two from the donor without levelling them, and the receipt records how
    much of the published spend the donor and the frame reach.
    """

    spec = parameters.get("published_fuels")
    if not isinstance(spec, Mapping):
        raise ValueError(
            f"{RECOMPOSE_FROM_REMAINDER_KIND} must declare published_fuels."
        )
    resource = str(spec.get("resource") or "")
    if resource not in UK_LCFS_VENDORED_RESOURCES:
        raise ValueError(f"{resource!r} is not a vendored lcfs_consumption resource.")
    period_value = int(spec["period_value"])
    classes = []
    for entry in spec.get("classes", ()):
        coicop, concept = str(entry["coicop"]), str(entry["concept"])
        column = str(entry["column"])
        if column not in UK_LCFS_FUEL_COLUMNS:
            raise ValueError(
                f"{column!r} is not a heating-fuel column the stage draws."
            )
        rows = vendored_rows(
            resource,
            concept=concept,
            period_type="calendar_year",
            period_value=period_value,
            geography_id="K02000001",
            dimensions={"coicop": coicop, "frequency": "annual"},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: expected one {coicop} row for {period_value}, "
                f"found {len(rows)}."
            )
        classes.append(
            {
                "coicop": coicop,
                "concept": concept,
                "column": column,
                "source_record_id": str(rows[0].get("source_record_id", "")),
                "value": float(rows[0]["value"]),
            }
        )
    if sorted(entry["column"] for entry in classes) != sorted(UK_LCFS_FUEL_COLUMNS):
        raise ValueError(
            f"{RECOMPOSE_FROM_REMAINDER_KIND} published_fuels must name "
            f"{sorted(UK_LCFS_FUEL_COLUMNS)} once each."
        )
    return {
        "resource": resource,
        "period_value": period_value,
        "classes": classes,
        "total": float(sum(entry["value"] for entry in classes)),
    }


def published_fuels_beside_draws(
    published: Mapping[str, Any],
    energy_split: Mapping[str, Any],
    frame_totals: Mapping[str, float],
) -> dict[str, Any]:
    """The published heating fuels beside the donor's and the frame's.

    Report-only: the donor's figure is its weighted annual diary total at
    diary prices, before the donor is carried to the base year, so it reads
    how much of ONS 04.5.3 and 04.5.4 the survey captures; the frame's is the
    stage's draws at the stage's weights, at calendar-2024 prices. ONS's
    04.5.3 leaves out the bottled gas the liquid column holds, so the liquid
    fuels' donor figure is also given without it (``UK_LCFS_FUEL_PARTS``).
    """

    classes = []
    for entry in published["classes"]:
        value = float(entry["value"])
        donor = float(energy_split[entry["column"]])
        frame = float(frame_totals[entry["column"]])
        row = {
            **entry,
            "donor_at_diary_prices": donor,
            "donor_over_published": donor / value if value > 0 else None,
            "frame": frame,
            "frame_over_published": frame / value if value > 0 else None,
        }
        if entry["column"] == "liquid_fuel_consumption" and "bottled_gas" in (
            energy_split
        ):
            bottled = float(energy_split["bottled_gas"])
            row["donor_bottled_gas_at_diary_prices"] = bottled
            row["donor_without_bottled_gas_over_published"] = (
                (donor - bottled) / value if value > 0 else None
            )
            row["concept_note"] = (
                "COICOP puts liquefied hydrocarbons under 04.5.2 (gas), so ONS's "
                "04.5.3 leaves out the bottled gas this column holds (María's "
                "ruling of 2026-10-22)."
            )
        classes.append(row)
    return {**published, "classes": classes}


def recompose_parent_totals(
    household_draws: pd.DataFrame,
    *,
    drawn: pd.DataFrame,
    parameters: Mapping[str, Any],
    weights: Sequence[float],
    donor_support: Mapping[str, tuple[float, float]],
    uprating_factors: Mapping[str, float] | None = None,
    donor_division_means: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Write each recomposed total around the levelled parts the stage set.

    A total keeps its own chain draw less the chain's draws of the parts the
    stage re-levels (``drawn``, kept before any step touched them; floored at
    zero where the parts outgrow the total), and adds the levelled parts back,
    so it always contains the electricity, gas, petrol, diesel and priced bus
    fares at their published scale (microcosm#1113). The chain draws each part after, and
    conditional on, its total, so the remainder is the chain's own split; a
    remainder drawn as a chain target of its own over-drew the vehicle-purchase
    tail of transport (10 to 13 percent above the donor in-sample).

    The total is then held to the donor's realised maximum of it
    (``donor_support``, the declared ``maximum_total``; microcosm#1164), never
    below its parts: what it gives up comes out of the remainder, and the
    receipt counts the rows held and the weighted mass taken.

    Where the declaration levels the divisions' relative over-draw
    (``level_division_excess``), each division above the shared ratio of the
    spend the chain drew is brought down to it before the hold
    (:func:`division_excess_factors`; ``donor_division_means`` is then the
    donor's drawn spend, :func:`lcfs_donor_drawn_means`): a recomposed total
    by its remainder alone, any other division by its draw.
    """

    check_recomposed_parents(parameters)
    weight = np.asarray(weights, dtype=float)
    factors = dict(uprating_factors or {})
    result = household_draws.copy()
    parents: dict[str, Any] = {}
    split: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for parent, (subtracted, components) in UK_LCFS_RECOMPOSED_PARENTS.items():
        drawn_total = result[parent].to_numpy(dtype=float)
        drawn_parts = sum(drawn[column].to_numpy(dtype=float) for column in subtracted)
        rest = np.maximum(drawn_total - drawn_parts, 0.0)
        parts = sum(result[column].to_numpy(dtype=float) for column in components)
        split[parent] = (drawn_total, drawn_parts, rest, parts)
    levelling = None
    level_factors: dict[str, float] = {}
    if parameters.get("level_division_excess") is not None:
        if donor_division_means is None:
            raise ValueError(
                "Levelling the divisions' excess needs the donor's division means."
            )
        # The spend the chain drew: a recomposed total's remainder, any other
        # division's whole draw.
        frame_drawn = {
            division: float(
                np.dot(
                    weight,
                    split[division][2]
                    if division in split
                    else result[division].to_numpy(dtype=float),
                )
            )
            for division in UK_LCFS_COICOP_DIVISION_COLUMNS
        }
        levelling = division_excess_factors(
            frame_drawn, donor_division_means, households=float(weight.sum())
        )
        level_factors = {
            division: entry["factor"]
            for division, entry in levelling["divisions"].items()
        }
        for division, factor in level_factors.items():
            if division not in split and factor != 1.0:
                result[division] = result[division].to_numpy(dtype=float) * factor
    for parent, (subtracted, components) in UK_LCFS_RECOMPOSED_PARENTS.items():
        if parent not in donor_support:
            raise ValueError(f"{parent}: no donor support to hold the total to.")
        maximum = float(donor_support[parent][1])
        drawn_total, drawn_parts, rest, parts = split[parent]
        level_factor = float(level_factors.get(parent, 1.0))
        rest = rest * level_factor
        unheld = rest + parts
        total = np.maximum(np.minimum(unheld, maximum), parts)
        result[parent] = total
        # The parent was uprated by its own factor and each netted part by its
        # own (a drawn part the uprating never touched by 1.0), so the
        # remainder keeps the gap between them. Recorded, not corrected
        # (microcosm#1113, the #1167 re-review).
        parent_factor = float(factors.get(parent, 1.0))
        residue = 0.0
        part_factors: dict[str, float] = {}
        for column in subtracted:
            part_factor = float(factors.get(column, 1.0))
            part_factors[column] = part_factor
            residue += float(np.dot(weight, drawn[column].to_numpy(dtype=float))) * (
                parent_factor / part_factor - 1.0
            )
        parents[parent] = {
            "uprating_residue": {
                "parent_factor": parent_factor,
                "part_factors": part_factors,
                "weighted_residue_gbp": residue,
            },
            "drawn_subtracts": list(subtracted),
            "components": list(components),
            "levelling_factor": level_factor,
            "weighted_drawn_total": float(np.dot(weight, drawn_total)),
            "weighted_drawn_subtracts": float(np.dot(weight, drawn_parts)),
            "rows_floored": int((drawn_parts > drawn_total).sum()),
            "weighted_floored_mass": float(
                np.dot(weight, np.maximum(drawn_parts - drawn_total, 0.0))
            ),
            "weighted_remainder": float(np.dot(weight, rest)),
            "weighted_components": float(np.dot(weight, parts)),
            "maximum_total": UK_LCFS_RECOMPOSED_MAXIMUM_TOTAL,
            "rows_held_at_maximum": int((unheld > total).sum()),
            "weighted_held_mass": float(np.dot(weight, unheld - total)),
            "weighted_total": float(np.dot(weight, total)),
            "minimum_remainder": float(rest.min()) if rest.size else 0.0,
            "rows_below_components": int((total < parts).sum()),
        }
    receipt = {
        "operation": RECOMPOSE_FROM_REMAINDER_KIND,
        "parents": parents,
        "published_fuels": published_fuel_spend(parameters),
        **({} if levelling is None else {"division_levelling": levelling}),
    }
    return result, receipt


def division_excess_factors(
    frame_drawn: Mapping[str, float],
    donor_drawn_means: Mapping[str, float],
    *,
    households: float,
) -> dict[str, Any]:
    """Each division's factor bringing its relative over-draw to the shared ratio.

    ``frame_drawn`` is each division's weighted frame spend the chain drew (a
    recomposed total's remainder, any other division's whole draw), and
    ``donor_drawn_means`` the donor's mean of the same per household
    (:func:`lcfs_donor_drawn_means`). A division's ratio is the one over the
    other; the shared ratio is the divisions' together, measured once. A
    division above it comes down to it, and every other keeps its draw
    (``UK_LCFS_LEVEL_DIVISION_EXCESS``).
    """

    expected = {
        division: households * float(donor_drawn_means[division])
        for division in UK_LCFS_COICOP_DIVISION_COLUMNS
    }
    ratios = {
        division: (
            float(frame_drawn[division]) / expected[division]
            if expected[division] > 0
            else None
        )
        for division in UK_LCFS_COICOP_DIVISION_COLUMNS
    }
    total_expected = sum(expected.values())
    if not total_expected > 0:
        raise ValueError("The donor's divisions hold no spend to level against.")
    shared = sum(float(frame_drawn[division]) for division in expected) / (
        total_expected
    )
    above = {
        division
        for division, ratio in ratios.items()
        if ratio is not None and ratio > shared
    }
    divisions = {
        division: {
            "ratio": ratios[division],
            "factor": (
                shared * expected[division] / float(frame_drawn[division])
                if division in above
                else 1.0
            ),
        }
        for division in UK_LCFS_COICOP_DIVISION_COLUMNS
    }
    return {
        "rule": UK_LCFS_LEVEL_DIVISION_EXCESS,
        "basis": UK_LCFS_LEVEL_DIVISION_BASIS,
        "shared_ratio": shared,
        "divisions": divisions,
    }


def lcfs_recompose_from_remainder(
    stage: SourceStageSpec,
    household_draws: pd.DataFrame,
    *,
    drawn: pd.DataFrame,
    weights: Sequence[float],
    donor_support: Mapping[str, tuple[float, float]],
    uprating_factors: Mapping[str, float] | None = None,
    donor_division_means: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply the declared ``recompose_from_remainder`` step (none if undeclared)."""

    parameters = recompose_from_remainder_operation(stage)
    if parameters is None:
        return household_draws, None
    return recompose_parent_totals(
        household_draws,
        drawn=drawn,
        parameters=parameters,
        weights=weights,
        donor_support=donor_support,
        uprating_factors=uprating_factors,
        donor_division_means=donor_division_means,
    )


def lcfs_donor_drawn_means(donor: pd.DataFrame) -> dict[str, float]:
    """Each division's drawn spend per donor household, at the donor's weights.

    A recomposed total's is the division less the parts the stage levels,
    floored at zero per household as the frame's remainder is; any other
    division's is the division. The donor is the one the chain drew from.
    """

    weight = _numeric(donor["household_weight"]).to_numpy(dtype=float)
    mass = float(weight.sum())
    if not mass > 0:
        raise ValueError("The LCFS donor carries no weight.")
    means: dict[str, float] = {}
    for division in UK_LCFS_COICOP_DIVISION_COLUMNS:
        spend = _numeric(donor[division]).to_numpy(dtype=float)
        if division in UK_LCFS_RECOMPOSED_PARENTS:
            subtracted, _ = UK_LCFS_RECOMPOSED_PARENTS[division]
            parts = sum(
                _numeric(donor[column]).to_numpy(dtype=float) for column in subtracted
            )
            spend = np.maximum(spend - parts, 0.0)
        means[division] = float(np.dot(weight, spend)) / mass
    return means


@dataclass(frozen=True)
class LCFSEnergyPricing:
    """The declared prices, NEED margins, gas connection and level, with receipts."""

    prices: EnergyPrices
    margins: NeedMargins
    gas_connected: str
    connection_shares: Mapping[str, float | None] | None
    disconnect_rule: str
    level: Mapping[str, float]
    receipt: dict[str, Any]
    disconnect_seed: int = 0
    ceilings: Mapping[str, float] | None = None
    connect_rule: str | None = None
    connect_salt: str | None = None
    connect_n_estimators: int | None = None
    zero_kwh_cells_at_need_mean: Mapping[str, tuple[str, ...]] | None = None


def lcfs_energy_pricing(stage: SourceStageSpec) -> LCFSEnergyPricing | None:
    """Resolve the declared ``price_domestic_energy`` operation (none if undeclared)."""

    parameters = pricing_operation(stage)
    if parameters is None:
        return None
    columns = dict(parameters["columns"])
    if columns != {"electricity": "electricity_consumption", "gas": "gas_consumption"}:
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} must price the two energy spend columns."
        )
    margins_resource = str(
        parameters.get("margins_resource") or UK_NEED_ENERGY_FACTS_RESOURCE
    )
    if "margins_period_value" not in parameters:
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} declares no margins_period_value (the "
            "NEED consumption year the shape comes from)."
        )
    prices, prices_receipt = qep_prices(parameters)
    margins = need_margins_from_facts(
        margins_resource, period_value=int(parameters["margins_period_value"])
    )
    level, level_receipt = published_energy_level(parameters)
    gas_connected = str(parameters.get("gas_connected") or "")
    if gas_connected == GAS_CONNECTED_PUBLISHED_METER_SHARE:
        shares, connection_receipt = published_gas_connected_shares(parameters)
    elif gas_connected == GAS_CONNECTED_POSITIVE_SPEND:
        shares, connection_receipt = None, {"rule": GAS_CONNECTED_POSITIVE_SPEND}
    else:
        raise ValueError(
            f"gas_connected must be {GAS_CONNECTED_POSITIVE_SPEND!r} or "
            f"{GAS_CONNECTED_PUBLISHED_METER_SHARE!r}, not {gas_connected!r}."
        )
    disconnect_rule = str(
        parameters.get("disconnect_rule") or DISCONNECT_LOWEST_DRAWN_GAS_FIRST
    )
    disconnect_seed = parameters.get("seed", 0)
    if disconnect_rule == DISCONNECT_IDENTITY_UNIFORM_ORDER and not (
        isinstance(disconnect_seed, int) and disconnect_seed >= 0
    ):
        raise ValueError(
            f"{PRICE_DOMESTIC_ENERGY_KIND} with disconnect_rule "
            f"{DISCONNECT_IDENTITY_UNIFORM_ORDER!r} must declare a non-negative "
            "integer seed."
        )
    ceilings = declared_household_kwh_ceilings(parameters)
    held_cells = declared_zero_kwh_cells_at_need_mean(parameters)
    connect_rule = parameters.get("connect_rule")
    connect_salt = parameters.get("connect_salt")
    connect_trees = parameters.get("connect_n_estimators")
    if connect_rule is not None:
        if connect_rule not in CONNECT_RULES:
            raise ValueError(f"connect_rule must be one of {CONNECT_RULES!r}.")
        if gas_connected != GAS_CONNECTED_PUBLISHED_METER_SHARE:
            raise ValueError(
                "connect_rule needs gas_connected: "
                f"{GAS_CONNECTED_PUBLISHED_METER_SHARE}."
            )
        if not isinstance(connect_salt, str) or not connect_salt:
            raise ValueError("connect_rule needs a declared connect_salt.")
        if not isinstance(connect_trees, int) or connect_trees < 1:
            raise ValueError("connect_rule needs a positive connect_n_estimators.")
    return LCFSEnergyPricing(
        prices=prices,
        margins=margins,
        gas_connected=gas_connected,
        connection_shares=shares,
        disconnect_rule=disconnect_rule,
        disconnect_seed=int(disconnect_seed),
        level=level,
        ceilings=ceilings,
        connect_rule=None if connect_rule is None else str(connect_rule),
        connect_salt=None if connect_rule is None else str(connect_salt),
        connect_n_estimators=None if connect_rule is None else int(connect_trees),
        zero_kwh_cells_at_need_mean=held_cells,
        receipt={
            **prices_receipt,
            "need_margins": margins.receipt,
            "level": level_receipt,
            "gas_connection": connection_receipt,
            "household_kwh_ceiling": ceilings,
            "connect_rule": connect_rule,
            "zero_kwh_cells_at_need_mean": None
            if held_cells is None
            else {margin: list(names) for margin, names in held_cells.items()},
        },
    )


@dataclass(frozen=True)
class GasPayerDraws:
    """Each recipient's probability of paying for gas and its gas if it does."""

    probability: np.ndarray
    positive_spend: np.ndarray
    receipt: dict[str, Any]
    fit_weight_records: tuple[FitWeightRecord, ...]


def gas_payer_draws(
    *,
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    household_ids: np.ndarray,
    energy: LCFSEnergyPricing,
) -> GasPayerDraws:
    """What the connection walk needs to connect a household the draw left at zero.

    Two weighted fits on the chain's predictors (microcosm#1164). A gradient
    boosted classifier of whether the donor pays for gas gives each recipient
    the probability that a donor like it does, which orders the walk; a
    regime-gated QRF of the donor's gas spend draws, from its positive forest
    at quantiles keyed on the household id, the gas such a donor pays, which a
    household the walk connects is given. The donor's gas is the labelled
    split's, so households heating with oil, bottled gas or solid fuel count
    as non-payers.
    """

    from sklearn.ensemble import HistGradientBoostingClassifier

    from microcosm.fit.qrf import Regime, RegimeGatedQRF

    target = "gas_consumption"
    seed = energy.disconnect_seed
    donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
        donor, recipient
    )
    donor_weight = donor_encoded["household_weight"].to_numpy(dtype=float)
    gas = donor_encoded[target].to_numpy(dtype=float)
    payers = gas > 0
    if payers.all() or not payers.any():
        raise ValueError(
            "the donor's gas spend has no zero-gas or no gas-paying household to "
            "learn who pays for gas from."
        )
    features = recipient_encoded[list(predictors)].reset_index(drop=True)
    classifier = HistGradientBoostingClassifier(random_state=seed)
    classifier.fit(
        donor_encoded[list(predictors)].to_numpy(dtype=float),
        payers.astype(int),
        sample_weight=donor_weight,
    )
    probability = classifier.predict_proba(features.to_numpy(dtype=float))[
        :, list(classifier.classes_).index(1)
    ]
    train = donor_encoded.loc[payers, [*predictors, "household_weight"]]
    train = train.reset_index(drop=True)
    train[target] = gas[payers]
    fitted = RegimeGatedQRF(
        n_estimators=int(energy.connect_n_estimators or 0), seed=seed
    ).fit(train, list(predictors), [target], weights="household_weight")
    regime = str(fitted.regimes()[target])
    if regime != Regime.POSITIVE_ONLY:
        raise ValueError(
            f"the donor's positive gas spend fitted regime {regime!r}, not "
            f"{Regime.POSITIVE_ONLY!r}."
        )
    positive = fitted.predict_positive_from_uniforms(
        features,
        quantiles={
            target: stable_identity_uniforms(
                np.asarray(household_ids),
                seed=seed,
                salt=f"{energy.connect_salt}:{target}",
            )
        },
    )[target].to_numpy(dtype=float)
    receipt = {
        "rule": energy.connect_rule,
        "salt": energy.connect_salt,
        "seed": seed,
        "n_estimators": energy.connect_n_estimators,
        "amount_regime": regime,
        "training_rows": rounded_count(int(payers.sum())),
        "donor_gas_payer_share": float(donor_weight[payers].sum())
        / float(donor_weight.sum()),
    }
    records = (
        FitWeightRecord(f"{UK_LCFS_CONSUMPTION_FIT_NAME}:gas_payer", EXPLICIT_KIND),
        FitWeightRecord(
            f"{UK_LCFS_CONSUMPTION_FIT_NAME}:gas_payer_amount", fitted.weight_kind
        ),
    )
    return GasPayerDraws(probability, positive, receipt, records)


def energy_spend_to_kwh(
    table: pd.DataFrame, *, energy: LCFSEnergyPricing, region: np.ndarray
) -> pd.DataFrame:
    """Add ``electricity_kwh`` and ``gas_kwh`` at the declared regional average prices."""

    result = table.copy()
    electricity = _numeric(result["electricity_consumption"]).to_numpy(dtype=float)
    gas = _numeric(result["gas_consumption"]).to_numpy(dtype=float)
    result[ELECTRICITY_KWH] = spend_to_kwh(
        electricity, frs_region=region, fuel="electricity", prices=energy.prices
    )
    result[GAS_KWH] = spend_to_kwh(
        gas, frs_region=region, fuel="gas", prices=energy.prices, connected=gas > 0
    )
    return result


def energy_kwh_to_spend(
    table: pd.DataFrame, *, energy: LCFSEnergyPricing, region: np.ndarray
) -> pd.DataFrame:
    """Price the kWh columns back to spend (gas fixed cost where connected) and drop them."""

    result = table.copy()
    gas_kwh = result[GAS_KWH].to_numpy(dtype=float)
    result["electricity_consumption"] = kwh_to_spend(
        result[ELECTRICITY_KWH].to_numpy(dtype=float),
        frs_region=region,
        fuel="electricity",
        prices=energy.prices,
    )
    result["gas_consumption"] = kwh_to_spend(
        gas_kwh,
        frs_region=region,
        fuel="gas",
        prices=energy.prices,
        connected=gas_kwh > 0,
    )
    return result.drop(columns=[ELECTRICITY_KWH, GAS_KWH])


def rake_recipient_energy(
    household_draws: pd.DataFrame,
    *,
    energy: LCFSEnergyPricing,
    region: np.ndarray,
    income: np.ndarray,
    tenure: np.ndarray,
    accommodation: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    identity: np.ndarray | None = None,
    gas_payers: GasPayerDraws | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Price the drawn spend to kWh, impose the published gas connection, rake the
    NEED shape at the DESNZ level, price back.

    ``identity`` (the household ids, row-aligned) keys the disconnection walk
    when the declared rule is ``identity_uniform_order``, and the connection
    walk. ``gas_payers`` (the declared ``connect_rule``) orders the households
    a region below its published share connects and gives each its gas."""

    in_kwh = energy_spend_to_kwh(household_draws, energy=energy, region=region)
    drawn_gas = in_kwh[GAS_KWH].to_numpy(dtype=float)
    weight_values = np.asarray(weights, dtype=float)
    if energy.gas_connected == GAS_CONNECTED_PUBLISHED_METER_SHARE:
        if (energy.connect_rule is None) != (gas_payers is None):
            raise ValueError(
                "gas_payers must be given exactly when the stage declares a "
                "connect_rule."
            )
        connected, connection_receipt = impose_gas_connection(
            drawn_gas,
            frs_region=region,
            weights=weight_values,
            shares=energy.connection_shares or {},
            disconnect_rule=energy.disconnect_rule,
            identity=identity,
            seed=energy.disconnect_seed,
            connect_probability=None if gas_payers is None else gas_payers.probability,
            connect_salt=energy.connect_salt,
        )
        in_kwh.loc[~connected, GAS_KWH] = 0.0
        if gas_payers is not None:
            added = connected & ~(drawn_gas > 0)
            in_kwh.loc[added, GAS_KWH] = spend_to_kwh(
                gas_payers.positive_spend[added],
                frs_region=np.asarray(region)[added],
                fuel="gas",
                prices=energy.prices,
                connected=np.ones(int(added.sum()), dtype=bool),
            )
            connection_receipt["gas_payers"] = dict(gas_payers.receipt)
    else:
        connected = drawn_gas > 0
        share = float(np.dot(connected, weight_values)) / max(
            float(weight_values.sum()), 1e-12
        )
        connection_receipt = {"share_before": share, "share_after": share}
    raked, receipt = rake_energy_kwh(
        in_kwh,
        margins=energy.margins,
        frs_region=region,
        income=income,
        weights=weight_values,
        iterations=iterations,
        tenure=tenure,
        accommodation=accommodation,
        use_region_margin=True,
        gas_connected=connected,
        level=energy.level,
        ceilings=energy.ceilings,
        zero_kwh_cells_at_need_mean=energy.zero_kwh_cells_at_need_mean,
    )
    receipt["gas_connection"] = {"rule": energy.gas_connected, **connection_receipt}
    receipt["gas_connected_share"] = float(
        np.dot(raked[GAS_KWH].to_numpy(dtype=float) > 0, weight_values)
        / max(float(weight_values.sum()), 1e-12)
    )
    return energy_kwh_to_spend(raked, energy=energy, region=region), receipt


def _energy_rake_operation(stage: SourceStageSpec, *, margin: str):
    """The declared energy IPF whose margins name ``margin`` (never positional)."""

    for operation in stage.operations:
        if operation.kind == "iterative_proportional_fit" and (
            "electricity_consumption" in operation.parameters.get("columns", ())
            and margin in [str(m) for m in operation.parameters.get("margins", ())]
        ):
            return operation
    return None


def _donor_energy_rake_iterations(stage: SourceStageSpec) -> int:
    """The declared donor-side energy IPF (the one on the gross income band)."""

    operation = _energy_rake_operation(stage, margin="gross_income_band")
    return 1 if operation is None else int(operation.parameters.get("iterations", 1))


def _recipient_energy_rake_iterations(stage: SourceStageSpec) -> int:
    """The declared post-imputation energy IPF (the one with the region margin).

    Refuses a stage without that IPF or its declared sweep count instead of
    falling back to a default: the energy_rake gate reads the residual's
    convergence over the last sweeps of exactly the declared count.
    """

    operation = _energy_rake_operation(stage, margin="region")
    if operation is None or "iterations" not in operation.parameters:
        raise ValueError(
            f"{stage.stage}: the post-imputation energy IPF (the one with the "
            "region margin) and its declared iterations are required."
        )
    return int(operation.parameters["iterations"])


def support_clip_exempt(stage: SourceStageSpec) -> set[str]:
    """Columns the declared ``support_clip`` exempts (raked columns keep their level)."""

    for operation in stage.operations:
        if operation.kind == "support_clip":
            declared = operation.parameters.get("exempt")
            if declared:
                return {str(column) for column in declared}
    return set(UK_LCFS_DEFAULT_SUPPORT_CLIP_EXEMPT)


def bus_pricing_operation(stage: SourceStageSpec) -> Mapping[str, Any] | None:
    """The stage's declared ``price_bus_journeys`` parameters, if any."""

    for operation in stage.operations:
        if operation.kind == PRICE_BUS_JOURNEYS_KIND:
            return {"kind": operation.kind, **operation.parameters}
    return None


def lcfs_bus_fare_pricing(
    stage: SourceStageSpec, household_draws: pd.DataFrame, frame: Frame
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Replace the priced regions' bus fares with journeys times the published yield.

    The chain's raw draw (clipped to donor support) stands where no area
    prices (Wales); everywhere else the emitted value is the sum over the
    household's non-eligible persons of their ``nts_bus_travel`` journeys
    times boardings per resident trip times the yield per fare-paying
    boarding (microcosm#930). The chain conditioned later targets on the raw
    draw, which the receipt states.
    """

    parameters = bus_pricing_operation(stage)
    if parameters is None:
        return household_draws, None
    column = str(parameters["column"])
    if column not in household_draws:
        raise ValueError(
            f"price_bus_journeys names {column!r}, which the chain did not draw."
        )
    prices = bus_fare_prices(parameters, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    trips_columns = {
        str(k): str(v) for k, v in dict(parameters["trips_columns"]).items()
    }
    household = frame.table("household")
    fares, priced, receipt = price_bus_journeys(
        frame.table("person"),
        household,
        prices=prices,
        trips_columns=trips_columns,
        eligibility_column=str(parameters["eligibility_column"]),
        household_weights=frame.weights_for("household").values,
        raw_household_fares=household_draws[column].to_numpy(dtype=float),
        region_boardings=english_region_boardings(
            parameters, allowed_resources=UK_LCFS_VENDORED_RESOURCES
        ),
    )
    priced_draws = household_draws.copy()
    priced_draws[column] = fares
    return priced_draws, {
        "operation": PRICE_BUS_JOURNEYS_KIND,
        "column": column,
        "chain_conditioned_on": "raw_draw",
        "households_priced": int(priced.sum()),
        **prices.receipt,
        **receipt,
    }


def lcfs_donor_uprating(
    stage: SourceStageSpec,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """The stage's declared donor uprating factors and receipt (none if undeclared)."""

    parameters = uprating_operation(stage)
    if parameters is None:
        return {}, None
    declared = {
        str(spec.get("resource"))
        for spec in parameters["columns"].values()
        if spec.get("resource")
    }
    if declared - {UK_LCFS_ROAD_FUEL_RESOURCE}:
        raise ValueError(
            "lcfs_consumption uprating may only read "
            f"{UK_LCFS_ROAD_FUEL_RESOURCE!r}; declared {sorted(declared)}."
        )
    return donor_uprating_factors(parameters)


def lcfs_ice_share(stage: SourceStageSpec) -> tuple[float, dict[str, Any]]:
    """The declared fuel-buyer share of licensed cars and its receipt.

    Read from the stage's ``assign_binary_from_rate`` declaration for
    ``has_fuel_consumption``: the vendored DfT VEH1103 licensed-car stock at
    the declared period and geography, one minus the zero-emission share.
    :func:`assign_recipient_has_fuel` draws each vehicle at it.
    """

    parameters = _operation_parameters(
        stage, "assign_binary_from_rate", target="has_fuel_consumption"
    )
    return ice_share_from_licensed_cars(parameters)


def ice_share_from_licensed_cars(
    parameters: Mapping[str, Any],
) -> tuple[float, dict[str, Any]]:
    """1 - (zero-emission licensed cars / all licensed cars) from vendored rows."""

    resource = str(parameters.get("rate_resource") or "")
    if resource != UK_LCFS_LICENSED_CARS_RESOURCE:
        raise ValueError(
            "has_fuel_consumption rate must come from "
            f"{UK_LCFS_LICENSED_CARS_RESOURCE!r}, not {resource!r}."
        )
    rule = str(parameters.get("rate_rule") or "")
    if rule != UK_LCFS_ICE_SHARE_RULE:
        raise ValueError(f"unsupported has_fuel_consumption rate_rule {rule!r}.")
    period_value = int(parameters["period_value"])
    geography_id = str(parameters["geography_id"])
    zero_emission_types = tuple(
        str(fuel_type) for fuel_type in parameters.get("zero_emission_fuel_types", ())
    )
    if not zero_emission_types:
        raise ValueError("has_fuel_consumption declares no zero_emission_fuel_types.")

    def licensed(fuel_type: str) -> tuple[float, str]:
        rows = vendored_rows(
            resource,
            period_type="calendar_year",
            period_value=period_value,
            geography_id=geography_id,
            dimensions={"fuel_type": fuel_type},
        )
        if len(rows) != 1:
            raise ValueError(
                f"{resource}: expected one {fuel_type!r} row for {period_value} "
                f"{geography_id}, found {len(rows)}."
            )
        return float(rows[0]["value"]), str(rows[0].get("source_record_id", ""))

    all_cars, all_record = licensed("all")
    zero_emission = {
        fuel_type: licensed(fuel_type) for fuel_type in zero_emission_types
    }
    if not np.isfinite(all_cars) or all_cars <= 0:
        raise ValueError(f"{resource}: licensed-car total must be positive.")
    zero_total = sum(count for count, _ in zero_emission.values())
    rate = 1.0 - zero_total / all_cars
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"{resource}: fuel-buyer share {rate} is outside (0, 1].")
    receipt = {
        "resource": resource,
        "rule": rule,
        "period_value": period_value,
        "geography_id": geography_id,
        "licensed_cars": all_cars,
        "zero_emission_licensed_cars": {
            fuel_type: count for fuel_type, (count, _) in zero_emission.items()
        },
        "rate": float(rate),
        "source_record_ids": [
            all_record,
            *(record for _, record in zero_emission.values()),
        ],
    }
    return float(rate), receipt


def vehicle_count(values: pd.Series) -> pd.Series:
    """Round and clip a vehicle count to ``UK_LCFS_VEHICLE_COUNT_RANGE``."""

    low, high = UK_LCFS_VEHICLE_COUNT_RANGE
    return np.rint(_numeric(values)).clip(low, high).astype("int64")


def fuel_flag_evidence(
    donor: pd.DataFrame,
    recipient: pd.DataFrame,
    *,
    recipient_weights: Sequence[float],
    ice_share_receipt: Mapping[str, Any],
    sources: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Weighted fuel-household shares on both sides of the imputation (donor design weights, frame prior weights).

    The flag is per household, so a zero-emission car in a household that
    also runs a petrol or diesel car keeps the household's fuel: the share
    of vehicles in fuel-free households sits below the per-car zero-emission
    share the draw uses, and the receipt records both (microcosm#1306
    review).

    The shares on one side difference into the households with and without
    a vehicle, crossed with fuel bought or not. So every share on that side
    is "<10" when any of those groups that is not empty rests on one to nine
    distinct positive-weight sources: the frame's source households
    (``sources``, not the copies it stacks of one), the donor's diaries.
    """

    donor_weights = _numeric(donor["household_weight"]).to_numpy(dtype=float)
    donor_vehicles = _numeric(donor["num_vehicles"]).to_numpy(dtype=float) > 0
    donor_fuel = (
        _numeric(donor["petrol_spending"]) + _numeric(donor["diesel_spending"])
    ).to_numpy(dtype=float) > 0
    donor_floor = _thinnest_group(
        donor_vehicles, donor_fuel, donor_weights, np.arange(len(donor))
    )
    weights = np.asarray(recipient_weights, dtype=float)
    recipient_vehicles = _numeric(recipient["num_vehicles"]).to_numpy(dtype=float) > 0
    vehicles = vehicle_count(recipient["num_vehicles"]).to_numpy(dtype=float)
    flagged = recipient["has_fuel_consumption"].to_numpy(dtype=bool)
    fuel_free = recipient_vehicles & ~flagged
    recipient_floor = _thinnest_group(
        recipient_vehicles,
        flagged,
        weights,
        np.arange(len(recipient)) if sources is None else np.asarray(sources),
    )
    return {
        "ice_share": dict(ice_share_receipt),
        "donor": {
            "households": rounded_count(len(donor)),
            "with_vehicles_share": small_cell_share(
                _weighted_share(donor_vehicles, donor_weights), donor_floor
            ),
            "positive_fuel_share": small_cell_share(
                _weighted_share(donor_fuel, donor_weights), donor_floor
            ),
            "positive_fuel_share_among_vehicle_households": small_cell_share(
                _weighted_share(
                    donor_fuel[donor_vehicles], donor_weights[donor_vehicles]
                ),
                donor_floor,
            ),
        },
        "recipient": {
            "households": int(len(recipient)),
            "with_vehicles_share": small_cell_share(
                _weighted_share(recipient_vehicles, weights), recipient_floor
            ),
            "flagged_share": small_cell_share(
                _weighted_share(flagged, weights), recipient_floor
            ),
            "flagged_share_among_vehicle_households": small_cell_share(
                _weighted_share(
                    flagged[recipient_vehicles], weights[recipient_vehicles]
                ),
                recipient_floor,
            ),
            "fuel_free_share_of_vehicles": small_cell_share(
                _weighted_share(
                    fuel_free[recipient_vehicles],
                    (weights * vehicles)[recipient_vehicles],
                ),
                recipient_floor,
            ),
        },
    }


def _thinnest_group(
    vehicles: np.ndarray,
    fuel: np.ndarray,
    weights: np.ndarray,
    sources: np.ndarray,
) -> int:
    """The fewest distinct positive-weight sources in a non-empty group of
    households with and without a vehicle, crossed with fuel or none."""

    positive = np.asarray(weights, dtype=float) > 0
    counts = [
        len(set(sources[positive & by_vehicle & by_fuel].tolist()))
        for by_vehicle in (vehicles, ~vehicles)
        for by_fuel in (fuel, ~fuel)
    ]
    return min((count for count in counts if count), default=0)


def _weighted_share(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    return float(np.sum(weights[np.asarray(mask, dtype=bool)]) / total)


#: How far a published LCFS total stored to one decimal place can sit from the
#: lines it sums by rounding alone, pounds a week.
UK_LCFS_TOTAL_ROUNDING = 0.05


def lcfs_donor_shares(
    stage: SourceStageSpec,
) -> dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """The excise shares the derive step declares, held to the module's.

    No declaration draws no share and leaves the chain its spend targets. A
    declaration that differs from ``UK_LCFS_EXCISE_SHARES`` in a share, its
    total or a diary line is refused: the code computes these. The loaded
    spec does not keep a mapping's order, so the chain's declared targets
    (``lcfs_chain_targets``) fix the order the shares are drawn in.
    """

    declared = _operation_parameters(stage, "derive").get("shares")
    if declared is None:
        return {}
    actual = (
        {
            name: (
                entry.get("total"),
                tuple(entry.get("numerator", ())),
                tuple(entry.get("denominator", ())),
            )
            for name, entry in declared.items()
            if isinstance(entry, Mapping)
        }
        if isinstance(declared, Mapping)
        else None
    )
    if actual != dict(UK_LCFS_EXCISE_SHARES):
        raise ValueError(
            f"{stage.stage}: the derive operation declares shares {declared!r}, "
            f"not the shares the stage draws: {dict(UK_LCFS_EXCISE_SHARES)!r}."
        )
    return dict(UK_LCFS_EXCISE_SHARES)


def lcfs_chain_targets(
    stage: SourceStageSpec, shares: Mapping[str, object]
) -> tuple[str, ...]:
    """The chain's declared targets, held to the spend targets then the shares."""

    declared = tuple(
        _operation_parameters(stage, "fit_weighted_qrf_chain").get("targets", ())
    )
    expected = (*UK_LCFS_CONSUMPTION_TARGET_COLUMNS, *shares)
    if declared != expected:
        raise ValueError(
            f"{stage.stage}: fit_weighted_qrf_chain declares targets {declared}, "
            f"not the spend targets followed by the declared shares: {expected}."
        )
    return expected


def donor_excise_shares(
    household: pd.DataFrame,
    shares: Mapping[str, tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Each declared share on the donor, from its weekly diary lines, and a receipt.

    Lines are read with refunds as zero, as the donor floor reads its totals,
    and a share is its lines over its denominator, zero where that is not
    positive and at most one. A ratio carries no price, so the uprating that
    moves the spend columns leaves it alone. ``household`` is the donor tab
    before the totals are renamed, annualised or uprated. The receipt gives,
    per share, the weighted share of the denominator its lines hold, the
    weighted share of donors with a positive line, the support of those
    donors as a band, how many donors the clip at one held, and, where the
    denominator is lines under a published total, the weighted share of
    donors whose published total sits beside those lines by more than its
    rounding ("<10" when one to nine donors do): a donor stage receipt is
    UKDS output (CD171 §5.2.1), so it gives no donor value and no small
    count. A share of the stakes an earlier share gives has no published
    total; its denominator is the stake lines it splits.
    """

    weights = _numeric(household["household_weight"]).to_numpy(dtype=float)
    mass = float(weights.sum())
    columns: dict[str, np.ndarray] = {}
    receipt: dict[str, Any] = {}
    for name, (total, numerator, denominator) in shares.items():
        source_total = (
            None
            if total in UK_LCFS_EXCISE_SPENDING_TOTALS
            else next(
                code
                for code, column in CONSUMPTION_VARIABLE_RENAMES.items()
                if column == total
            )
        )
        for code in (
            *numerator,
            *denominator,
            *([source_total] if source_total else []),
        ):
            if code not in household:
                raise ValueError(f"LCFS household donor is missing {code!r}.")

        def lines(codes: tuple[str, ...]) -> np.ndarray:
            return sum(
                _numeric(household[code]).clip(lower=0.0).to_numpy(dtype=float)
                for code in codes
            )

        top = lines(numerator)
        bottom = lines(denominator)
        ratio = np.divide(top, bottom, out=np.zeros(len(household)), where=bottom > 0)
        clipped = int((ratio > 1.0).sum())
        columns[name] = np.clip(ratio, 0.0, 1.0)
        positive = top > 0
        entry: dict[str, Any] = {
            "total": total,
            "numerator": list(numerator),
            "denominator": list(denominator),
            "share_of_denominator": float(weights @ top / (weights @ bottom))
            if weights @ bottom > 0
            else None,
            "positive_share": float(weights[positive].sum() / mass) if mass else None,
            "positive_support": support_band(int(positive.sum())),
            "clipped_at_one": extreme_count(clipped),
        }
        if source_total is not None and source_total not in denominator:
            published = _numeric(household[source_total]).to_numpy(dtype=float)
            beside = np.abs(published - bottom) > UK_LCFS_TOTAL_ROUNDING + 1e-9
            entry["total_beside_lines"] = small_cell_share(
                float(weights[beside].sum() / mass) if mass else 0.0,
                int(beside.sum()),
            )
        receipt[name] = entry
    return pd.DataFrame(columns, index=household.index), receipt


def excise_spending(draws: pd.DataFrame) -> pd.DataFrame:
    """The excise spending the drawn shares give: scratch, on no table.

    Tobacco and off-trade alcohol split the drawn alcohol and tobacco total
    between them, so together they are that total; on-trade alcohol is its
    share of restaurants and hotels, and gambling stakes theirs of recreation.
    The National Lottery's stakes are their share of the stakes, and the
    betting and gaming stakes theirs of the rest.
    """

    tobacco = draws[UK_LCFS_TOBACCO_SHARE].to_numpy(dtype=float)
    alcohol_and_tobacco = draws["alcohol_and_tobacco_consumption"].to_numpy(dtype=float)
    on_trade = draws[UK_LCFS_ON_TRADE_ALCOHOL_SHARE].to_numpy(dtype=float)
    stakes = draws[UK_LCFS_GAMBLING_SHARE].to_numpy(dtype=float) * draws[
        "recreation_consumption"
    ].to_numpy(dtype=float)
    lottery = draws[UK_LCFS_NATIONAL_LOTTERY_SHARE].to_numpy(dtype=float)
    other_stakes = (1.0 - lottery) * stakes
    return pd.DataFrame(
        {
            UK_LCFS_TOBACCO_SPENDING: tobacco * alcohol_and_tobacco,
            UK_LCFS_OFF_TRADE_ALCOHOL_SPENDING: (1.0 - tobacco) * alcohol_and_tobacco,
            UK_LCFS_ON_TRADE_ALCOHOL_SPENDING: on_trade
            * draws["restaurants_and_hotels_consumption"].to_numpy(dtype=float),
            UK_LCFS_GAMBLING_SPENDING: stakes,
            UK_LCFS_NATIONAL_LOTTERY_SPENDING: lottery * stakes,
            UK_LCFS_OTHER_GAMBLING_SPENDING: other_stakes,
            UK_LCFS_BETTING_AND_GAMING_SPENDING: draws[
                UK_LCFS_BETTING_AND_GAMING_SHARE
            ].to_numpy(dtype=float)
            * other_stakes,
        },
        index=draws.index,
    )


def clip_excise_shares(
    draws: pd.DataFrame,
    shares: Mapping[str, tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Hold each drawn share to [0, 1], counting the draws held.

    The chain draws a share from donor values, which lie in the interval, so
    the count is zero unless the chain stops doing so.
    """

    held = draws.copy()
    outside: dict[str, int | str] = {}
    for name in shares:
        values = held[name].to_numpy(dtype=float)
        outside[name] = extreme_count(int(((values < 0.0) | (values > 1.0)).sum()))
        held[name] = np.clip(values, 0.0, 1.0)
    return held, outside


def excise_shares_receipt(
    draws: pd.DataFrame,
    *,
    shares: Mapping[str, tuple[str, tuple[str, ...], tuple[str, ...]]],
    weights: np.ndarray,
    donor_receipt: Mapping[str, Any],
    outside: Mapping[str, int | str],
) -> dict[str, Any]:
    """The shares on the frame beside the donor's: report-only (microcosm#1359).

    Per share, at prior weights: the weighted share of its total the frame's
    part holds and the weighted share of households with a positive part,
    beside the donor's ``donor_excise_shares`` receipt and the draws the clip
    held.
    """

    weights = np.asarray(weights, dtype=float)
    mass = float(weights.sum())
    spending = excise_spending(draws)
    parts = {
        UK_LCFS_TOBACCO_SHARE: spending[UK_LCFS_TOBACCO_SPENDING],
        UK_LCFS_ON_TRADE_ALCOHOL_SHARE: spending[UK_LCFS_ON_TRADE_ALCOHOL_SPENDING],
        UK_LCFS_GAMBLING_SHARE: spending[UK_LCFS_GAMBLING_SPENDING],
        UK_LCFS_NATIONAL_LOTTERY_SHARE: spending[UK_LCFS_NATIONAL_LOTTERY_SPENDING],
        UK_LCFS_BETTING_AND_GAMING_SHARE: spending[UK_LCFS_BETTING_AND_GAMING_SPENDING],
    }
    receipt: dict[str, Any] = {}
    for name, (total, _numerator, _denominator) in shares.items():
        part = parts[name].to_numpy(dtype=float)
        whole = (spending if total in UK_LCFS_EXCISE_SPENDING_TOTALS else draws)[
            total
        ].to_numpy(dtype=float)
        receipt[name] = {
            "total": total,
            "frame": {
                "share_of_total": float(weights @ part / (weights @ whole))
                if weights @ whole > 0
                else None,
                "positive_share": float(weights[part > 0].sum() / mass)
                if mass
                else None,
                "draws_outside_unit_interval": outside[name],
            },
            "donor": dict(donor_receipt[name]),
        }
    return receipt


def clean_lcfs_consumption_table(
    lcfs_person: pd.DataFrame,
    lcfs_household: pd.DataFrame,
    *,
    uprating: Mapping[str, float] | None = None,
    energy: LCFSEnergyPricing | None = None,
    donor_rake_iterations: int = 1,
    shares: Mapping[str, tuple[str, tuple[str, ...], tuple[str, ...]]] | None = None,
) -> pd.DataFrame:
    """Return the LCFS donor table with annualized consumption variables.

    ``uprating`` maps donor columns to the declared factors that move the
    2024-25 diary to the base year (``uprate_donor_columns``): the spend
    columns a quarter back to calendar-2024 prices, the income predictors by
    one (the diary year is the FRS 2024-25 year; microcosm#1164). It is applied after annualisation and before the
    donor-side NEED rake, so the income bands and the support-clip ranges see
    base-year values.
    ``energy`` (the declared ``price_domestic_energy``) converts the diary's
    energy spend to kWh at the calendar-2024 regional average prices paid, rakes
    the income margin to the NEED means (shape only, no level) and prices back; without it the energy columns
    are the raw diary spend.
    ``shares`` (the derive step's declared excise shares, microcosm#1359) adds
    each share as a donor column for the chain, from the diary lines before
    any of this, and its receipt under ``donor.attrs``.
    """

    person = _lowercase(lcfs_person).rename(columns=PERSON_LCFS_RENAMES)
    household = _lowercase(lcfs_household).rename(columns=HOUSEHOLD_LCFS_RENAMES)
    _require_columns(household, ("case", *HOUSEHOLD_LCFS_RENAMES.values()))
    _require_columns(person, ("case", *PERSON_LCFS_RENAMES.values()))
    household["region"] = mapped_lcfs_codes(
        household["region"], LCFS_REGIONS, column="gorx"
    )
    household["num_vehicles"] = vehicle_count(household["num_vehicles"])
    household["tenure_type"] = mapped_lcfs_codes(
        _lowercase(lcfs_household)["a122"], LCFS_TENURE_MAP, column="a122"
    )
    household["accommodation_type"] = mapped_lcfs_codes(
        _lowercase(lcfs_household)["a121"], LCFS_ACCOMM_MAP, column="a121"
    )
    household = derive_energy_from_lcfs(household)
    energy_split = donor_energy_split_receipt(household)
    household[UK_LCFS_PREPAYMENT_METER_COLUMN] = prepayment_meter_from_lcfs(household)
    shares = dict(shares or {})
    share_columns, share_receipt = donor_excise_shares(household, shares)
    for name in shares:
        household[name] = share_columns[name]
    household = household.rename(columns=CONSUMPTION_VARIABLE_RENAMES)
    for code in BUS_FARE_LCFS_CODES:
        if code not in household:
            raise ValueError(f"LCFS household donor is missing {code!r}.")
    household["bus_fare_spending"] = sum(
        _numeric(household[code]) for code in BUS_FARE_LCFS_CODES
    )
    annualize = [
        *CONSUMPTION_VARIABLE_RENAMES.values(),
        "bus_fare_spending",
        "hbai_household_net_income",
        "household_gross_income",
        "electricity_consumption",
        "gas_consumption",
        *UK_LCFS_FUEL_COLUMNS,
    ]
    for column in annualize:
        household[column] = _numeric(household[column]) * WEEKS_IN_YEAR
    for column in PERSON_LCFS_RENAMES.values():
        totals = person.groupby("case")[column].sum()
        household[column] = household["case"].map(totals).fillna(0.0) * WEEKS_IN_YEAR
    household["household_weight"] = _numeric(household["household_weight"]) * 1_000
    if uprating:
        household = apply_donor_uprating(household, uprating)
    if energy is not None:
        region = household["region"].astype(str).to_numpy()
        in_kwh = energy_spend_to_kwh(household, energy=energy, region=region)
        raked, _receipt = rake_energy_kwh(
            in_kwh,
            margins=energy.margins,
            frs_region=region,
            income=household["household_gross_income"].to_numpy(dtype=float),
            weights=None,
            iterations=donor_rake_iterations,
            gas_connected=in_kwh[GAS_KWH].to_numpy(dtype=float) > 0,
        )
        household = energy_kwh_to_spend(raked, energy=energy, region=region)
    household["domestic_energy_consumption"] = (
        household["electricity_consumption"] + household["gas_consumption"]
    )
    donor = household[
        [
            *UK_LCFS_CONSUMPTION_PREDICTORS,
            *UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
            *shares,
            UK_LCFS_PREPAYMENT_METER_COLUMN,
            "household_gross_income",
            "household_weight",
        ]
    ]
    # Every donor is kept or the stage stops: a silent drop would lose donors
    # whenever a survey year recodes a variable (microcosm#1164).
    missing = sorted(donor.columns[donor.isna().any()])
    if missing:
        raise ValueError(f"LCFS donor has missing values in {missing}.")
    donor.attrs[UK_LCFS_ENERGY_SPLIT_RECEIPT] = energy_split
    if shares:
        donor.attrs[UK_LCFS_EXCISE_SHARES_RECEIPT] = share_receipt
    return donor


def donor_energy_split_receipt(household: pd.DataFrame) -> dict[str, Any]:
    """The donor's fuel, light and power by fuel: weighted annual diary totals.

    Electricity and gas are what ``derive_energy_from_lcfs`` gives the chain;
    the other fuels stay in the housing total. Their sum against ``p537``
    shows how much of the published definition the labelled parts reach
    (microcosm#1164), split into what reading refunds as zero adds
    (``clipping_over_fuel_light_and_power``) and what the parts, unclipped,
    differ from p537 by (``unclipped_over_fuel_light_and_power``), with the
    weighted share of donors whose unclipped parts differ from their p537 by
    more than a penny and by more than 50p a week (``"<10"`` when fewer than
    ten donors do). Totals over every donor, at diary prices, and the account
    identity shares the split checked, floored to two decimals.
    """

    weight = _numeric(household["household_weight"]).to_numpy(dtype=float) * 1_000
    annual = weight * WEEKS_IN_YEAR

    def total(values: pd.Series | np.ndarray) -> float:
        return float(np.dot(annual, np.asarray(values, dtype=float)))

    def code(name: str) -> pd.Series:
        return _numeric(household[name])

    electricity = total(_numeric(household["electricity_consumption"]))
    gas = total(_numeric(household["gas_consumption"]))
    other = total(_numeric(household[UK_LCFS_OTHER_FUELS_COLUMN]))
    p537 = code("p537")
    fuel_light_power = total(p537)
    unclipped = code(UK_LCFS_ELECTRICITY_ACCOUNT[0]) + code(UK_LCFS_GAS_ACCOUNT[0])
    for name in (*UK_LCFS_ELECTRICITY_PARTS, *UK_LCFS_GAS_PARTS):
        if name not in {
            parts_code
            for _, _, parts in UK_LCFS_COMBINED_PAYMENTS
            for parts_code in parts
        }:
            unclipped = unclipped + code(name)
    for combined, rebate, _parts in UK_LCFS_COMBINED_PAYMENTS:
        unclipped = (
            unclipped + code(combined) - (0.0 if rebate is None else code(rebate))
        )
    for name in UK_LCFS_OTHER_FUEL_PARTS:
        unclipped = unclipped + code(name)
    parts = (
        _numeric(household["electricity_consumption"])
        + _numeric(household["gas_consumption"])
        + _numeric(household[UK_LCFS_OTHER_FUELS_COLUMN])
    )
    gap = (unclipped - p537).abs().to_numpy()
    mass = float(weight.sum())

    def differing(threshold: float) -> float | str | None:
        rows = gap > threshold
        if mass <= 0:
            return None
        return small_cell_share(float(weight[rows].sum()) / mass, int(rows.sum()))

    def over(value: float) -> float | None:
        return value / fuel_light_power if fuel_light_power > 0 else None

    return {
        "electricity": electricity,
        "gas": gas,
        "other_fuels": other,
        **{
            column: total(_numeric(household[column]))
            for column in UK_LCFS_FUEL_COLUMNS
        },
        **(
            {"bottled_gas": total(_numeric(household[UK_LCFS_BOTTLED_GAS_COLUMN]))}
            if UK_LCFS_BOTTLED_GAS_COLUMN in household
            else {}
        ),
        "fuel_light_and_power": fuel_light_power,
        "parts_over_fuel_light_and_power": over(electricity + gas + other),
        "clipping_over_fuel_light_and_power": over(total(parts - unclipped)),
        "unclipped_over_fuel_light_and_power": over(total(unclipped)),
        "share_of_donors_whose_unclipped_parts_differ_from_p537": {
            "by_over_a_penny_a_week": differing(0.01),
            "by_over_50p_a_week": differing(0.50),
        },
        "electricity_share_of_electricity_and_gas": (
            electricity / (electricity + gas) if electricity + gas > 0 else None
        ),
        "account_identity_shares": dict(
            household.attrs.get(UK_LCFS_ACCOUNT_IDENTITY_RECEIPT) or {}
        ),
    }


def mapped_lcfs_codes(
    raw: pd.Series, mapping: Mapping[int, str], *, column: str
) -> pd.Series:
    """Map an LCFS code column, refusing any code the map does not name.

    Declared as ``unmapped_codes: refuse`` on the stage's derive step
    (microcosm#1164). The message names each unknown code, and blank or
    non-numeric entries, without counting rows: a build log is not an output
    table, and a count of a rare code would be a small cell if it were posted.
    """

    codes = pd.to_numeric(raw, errors="coerce")
    mapped = codes.map(mapping)
    unknown = mapped.isna().to_numpy()
    if unknown.any():
        bad = codes[unknown]
        labels = [f"{code:g}" for code in sorted(bad.dropna().unique())]
        if bad.isna().any():
            labels.append("blank or non-numeric")
        raise ValueError(
            f"LCFS {column} has code(s) the stage does not map: {', '.join(labels)}."
        )
    return mapped


def require_refused_unmapped_codes(stage: SourceStageSpec) -> None:
    """The derive step must declare the unknown-code refusal the code applies."""

    declared = _operation_parameters(stage, "derive").get("unmapped_codes")
    if declared != "refuse":
        raise ValueError(
            f"{stage.stage}: the derive operation must declare unmapped_codes: "
            f"refuse, which mapped_lcfs_codes applies; got {declared!r}."
        )


def donor_consumption_floor(stage: SourceStageSpec) -> float:
    """The declared floor of the donor's diary consumption columns."""

    parameters = _operation_parameters(stage, "derive")
    floor = parameters.get("floor")
    if floor is None:
        raise ValueError(
            f"{stage.stage}: the derive operation must declare 'floor', the "
            "value negative diary consumption is raised to on the donor."
        )
    value = float(floor)
    if not math.isfinite(value) or value != 0.0:
        raise ValueError(
            f"{stage.stage}: the declared donor consumption floor must be 0, "
            f"got {floor!r}."
        )
    return value


def floor_negative_donor_consumption(
    donor: pd.DataFrame, *, floor: float = 0.0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Raise negative diary consumption on the donor to ``floor`` (microcosm#1063 c9).

    An LCFS diary spend nets refunds against purchases, so a handful of donor
    rows carry a negative annual total, and hundreds of release rows inherited
    negative ``housing_water_and_electricity_consumption`` on the 2026-09-30
    build, because the support clip takes its floor from the donor's realised
    range. The release surface declares these columns non-negative, so the
    donor is floored before the imputation, the clip ranges and the rake see
    it. The receipt counts, per consumption column, the rows raised, in the
    bands of ``receipt_counts.extreme_count``; columns without a negative row
    are listed with zeros so the receipt names the whole surface. It records
    no donor value and no statistic of the raised rows, which are few: a
    stage receipt is UKDS output (CD171 §5.2.1).
    """

    floored = donor.copy()
    columns: dict[str, dict[str, int | str]] = {}
    raised = 0
    for column in UK_LCFS_DONOR_FLOORED_COLUMNS:
        if column not in floored.columns:
            raise ValueError(f"LCFS donor is missing consumption column {column!r}.")
        values = _numeric(floored[column]).to_numpy(dtype=float)
        negative = np.isfinite(values) & (values < floor)
        raised += int(negative.sum())
        columns[column] = {"rows_raised": extreme_count(int(negative.sum()))}
        if negative.any():
            floored[column] = np.where(negative, floor, values)
    receipt = {
        "floor": float(floor),
        "columns": columns,
        "rows_raised": extreme_count(raised),
        "remaining_negative_rows": int(
            sum(
                int((_numeric(floored[column]).to_numpy(dtype=float) < floor).sum())
                for column in columns
            )
        ),
    }
    return floored, receipt


def account_identity_shares(household: pd.DataFrame) -> dict[str, float | None]:
    """The share of each fuel's account payers whose account less rebate is exact.

    b489 must equal b175 less b178, and b490 b170 less b173, to the penny (the
    labels of Volume F Part 2 rows 506/507). The share is over households with
    any of the three amounts, unweighted and exact, for the check; none where
    no household has one. The split's receipt floors it to two decimals.
    """

    shares: dict[str, float | None] = {}
    for fuel, (net, paid, rebate) in (
        ("electricity", UK_LCFS_ELECTRICITY_ACCOUNT),
        ("gas", UK_LCFS_GAS_ACCOUNT),
    ):
        n, p, r = (_numeric(household[code]) for code in (net, paid, rebate))
        payers = (n != 0) | (p != 0) | (r != 0)
        holds = (n - (p - r)).abs() <= 0.01
        shares[fuel] = float(holds[payers].mean()) if payers.any() else None
    return shares


def derive_energy_from_lcfs(household: pd.DataFrame) -> pd.DataFrame:
    """Split LCFS fuel, light and power into electricity and gas (weekly amounts).

    Each fuel is the sum of its labelled parts (``UK_LCFS_ELECTRICITY_*``,
    ``UK_LCFS_GAS_*``; microcosm#1164): the last account payment less its
    rebate (b489, b490, which the donor must show equal to the payment less
    the rebate for nearly every account payer, ``account_identity_shares``),
    the prepayment-meter payment, the fuel's component of a combined utilities
    bill and of a combined key/meter payment, and the diary's second-dwelling
    account and slot-meter lines. What a combined payment leaves unsplit, net
    of the combined bill's rebate (b174), is shared by the household's own
    electricity share where it pays for both fuels, and otherwise by the
    donor's mean share of the households that do: a combined payment means
    the household pays for both, so reporting only one fuel elsewhere does not
    make the whole payment that fuel. A donor with no such household fails.
    Oil, bottled gas, solid fuels and heat (``UK_LCFS_OTHER_FUEL_PARTS``) are
    neither: they stay in the housing total. Negative amounts (refunds) are
    read as zero.

    The derivation this replaces read b226 as electricity paid by direct debit
    and b489/b490 as prepayment payments; the LCF labels make b226 the gas
    component of a combined bill and b489/b490 the electricity and gas account
    payments less rebates, so it gave combined-bill households their gas as
    electricity and folded other fuels into gas.
    """

    _require_columns(household, UK_LCFS_ENERGY_CODES)
    shares = account_identity_shares(household)
    for fuel_name, share_holding in shares.items():
        if (
            share_holding is not None
            and share_holding < UK_LCFS_ACCOUNT_IDENTITY_MINIMUM
        ):
            raise ValueError(
                f"LCFS {fuel_name} account less rebate equals the payment less the "
                f"rebate for under {floored_share(share_holding) + 0.01:.2f} of "
                f"account payers (at least {UK_LCFS_ACCOUNT_IDENTITY_MINIMUM} "
                "expected): the labels or the rebate's sign have moved."
            )

    def part(code: str) -> pd.Series:
        return _numeric(household[code]).clip(lower=0.0)

    def fuel(account: tuple[str, str, str], parts: tuple[str, ...]) -> pd.Series:
        total = part(account[0])
        for code in parts:
            total = total + part(code)
        return total

    electricity = fuel(UK_LCFS_ELECTRICITY_ACCOUNT, UK_LCFS_ELECTRICITY_PARTS)
    gas = fuel(UK_LCFS_GAS_ACCOUNT, UK_LCFS_GAS_PARTS)
    unsplit = pd.Series(0.0, index=household.index)
    for total, rebate, parts in UK_LCFS_COMBINED_PAYMENTS:
        paid = _numeric(household[total])
        if rebate is not None:
            paid = paid - _numeric(household[rebate])
        split = sum(part(code) for code in parts)
        unsplit = unsplit + (paid.clip(lower=0.0) - split).clip(lower=0.0)
    both = (electricity > 0) & (gas > 0)
    if not both.any():
        raise ValueError(
            "no LCFS donor pays for both electricity and gas, so an unsplit "
            "combined payment has no electricity share to take."
        )
    mean_share = float((electricity[both] / (electricity[both] + gas[both])).mean())
    share = np.where(
        both, electricity / (electricity + gas).where(both, 1.0), mean_share
    )
    result = household.copy()
    result["electricity_consumption"] = electricity + unsplit * share
    result["gas_consumption"] = gas + unsplit * (1.0 - share)
    result[UK_LCFS_OTHER_FUELS_COLUMN] = sum(
        part(code) for code in UK_LCFS_OTHER_FUEL_PARTS
    )
    for column, codes in UK_LCFS_FUEL_PARTS.items():
        result[column] = sum(part(code) for code in codes)
    result[UK_LCFS_BOTTLED_GAS_COLUMN] = sum(
        part(code) for code in UK_LCFS_BOTTLED_GAS_PARTS
    )
    result.attrs[UK_LCFS_ACCOUNT_IDENTITY_RECEIPT] = {
        fuel_name: None if share_holding is None else floored_share(share_holding)
        for fuel_name, share_holding in shares.items()
    }
    return result


def assign_recipient_has_fuel(frame: Frame, *, rate: float, seed: int) -> np.ndarray:
    """Flag the recipient households that buy road fuel.

    ``rate`` is the fuel-buyer share of licensed cars, one minus the
    zero-emission share. Each of a household's vehicles is zero-emission at
    that share, independently, so a household with ``n`` vehicles buys fuel
    unless all ``n`` are: it is flagged at ``1 - (1 - rate) ** n``, against a
    uniform keyed on its identity (microcosm#1278). A one-vehicle household
    keeps the draw it had when the whole household was drawn at ``rate``.
    """

    household = frame.table("household")
    if "num_vehicles" not in household:
        raise KeyError("recipient household table is missing 'num_vehicles'.")
    if not 0.0 < float(rate) <= 1.0:
        raise ValueError(f"fuel-buyer share {rate} is outside (0, 1].")
    draws = stable_identity_uniforms(
        household["household_id"].to_numpy(),
        seed=seed,
        salt="lcfs_has_fuel_consumption",
    )
    has_vehicle = (_numeric(household["num_vehicles"]) > 0).to_numpy()
    vehicles = np.maximum(vehicle_count(household["num_vehicles"]).to_numpy(), 1)
    buys_fuel = 1.0 - (1.0 - float(rate)) ** vehicles
    return has_vehicle & (np.asarray(draws, dtype=np.float64) < buys_fuel)


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize LCFS recipient predictors at household grain.

    ``household_gross_income``, the NEED band's income, is the HBAI net
    income with the engine's HBAI deductions added back (``hbai_gross_income``);
    the deductions are not kept.
    """

    materialized = engine.materialize(
        frame, UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    result = pd.DataFrame(index=household.index)
    for variable in UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS:
        if variable not in UK_HBAI_GROSS_INCOME_DEDUCTIONS:
            # The HBAI net income is read before the reset to zero and keeps
            # its predictor name, the donor's p389p column.
            predictor = (
                UK_HBAI_NET_INCOME_PREDICTOR
                if variable == UK_HBAI_NET_INCOME_VARIABLE
                else variable
            )
            result[predictor] = household_level(
                frame, engine, variable, materialized[variable]
            )
    result["household_gross_income"] = hbai_gross_income(frame, engine, materialized)
    age = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float)
    adult = age >= UK_LCFS_HOUSEHOLD_COUNT_ADULT_AGE
    for predictor, members in zip(
        UK_LCFS_HOUSEHOLD_COUNT_PREDICTORS, (adult, ~adult), strict=True
    ):
        counted = (
            pd.Series(members.astype(float))
            .groupby(person["person_household_id"].to_numpy())
            .sum()
        )
        result[predictor] = (
            counted.reindex(household["household_id"]).fillna(0.0).to_numpy()
        )
    for predictor in ("region", "tenure_type", "accommodation_type"):
        if predictor in household:
            result[predictor] = household[predictor].map(_enum_name).to_numpy()
    if "num_vehicles" not in household:
        raise KeyError(
            "recipient household table is missing 'num_vehicles' "
            "(the was_wealth draw the consumption QRF conditions on)."
        )
    result["num_vehicles"] = vehicle_count(household["num_vehicles"]).to_numpy()
    return result


def impute_lcfs_consumption(
    donor: pd.DataFrame,
    recipient_predictor_frame: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
    record_ids: Sequence[object] | np.ndarray,
    targets: Sequence[str] = UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
) -> UKLCFSConsumptionImputationResult:
    """Chain-draw the targets in the declared order.

    ``bus_fare_spending`` stays a chain target so later targets condition on
    the same raw draw; the emitted fares are then replaced by the priced
    value where an area prices (microcosm#930).

    Each draw reads uniforms keyed on the household's own id (``record_ids``,
    in recipient row order) and the target, never a positional stream, so a
    household's consumption moves only when its own predictors do.
    ``targets`` is the spend targets, optionally followed by the declared
    excise shares (microcosm#1359): a target appended after them leaves every
    earlier draw as it was.
    """

    targets = tuple(targets)
    if targets[: len(UK_LCFS_CONSUMPTION_TARGET_COLUMNS)] != (
        UK_LCFS_CONSUMPTION_TARGET_COLUMNS
    ):
        raise ValueError(
            "The LCFS chain draws its spend targets first, in their declared "
            f"order, then any excise shares; got {targets}."
        )

    from microcosm.fit import RegimeGatedQRF

    donor_encoded, recipient_encoded, predictors = _encode_consumption_predictors(
        donor, recipient_predictor_frame
    )
    ids = integer_record_keys(record_ids, label="LCFS record_ids")
    if ids.shape != (len(recipient_encoded),) or pd.Series(ids).duplicated().any():
        raise ValueError(
            "LCFS record_ids must name every recipient once, in recipient row order."
        )
    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    state = model.start_chain(
        donor_encoded,
        list(predictors),
        list(targets),
        weights="household_weight",
    )
    raw = pd.DataFrame(index=recipient_encoded.index)
    draws = pd.DataFrame(index=recipient_encoded.index)
    fit_records: list[FitWeightRecord] = []
    for target in targets:
        features = recipient_encoded.loc[:, list(predictors)]
        salt = f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}"
        result = model.fit_draw_next(
            donor_encoded,
            features,
            raw,
            state=state,
            weights="household_weight",
            quantiles=stable_identity_uniforms(ids, seed=seed, salt=salt),
            sign_uniforms=stable_identity_uniforms(ids, seed=seed, salt=f"{salt}:sign"),
        )
        raw[target] = result.raw_draw
        draws[target] = result.raw_draw
        fit_records.append(
            FitWeightRecord(
                f"{UK_LCFS_CONSUMPTION_FIT_NAME}:{target}", result.weight_kind
            )
        )
        state = result.state
    return UKLCFSConsumptionImputationResult(draws, tuple(fit_records))


def support_clip_to_donor(
    draws: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    exempt: set[str] | None = None,
) -> UKSupportClipResult:
    return support_clip_to_donor_with_receipt(
        draws,
        donor,
        columns=UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
        stage=UK_LCFS_CONSUMPTION_STAGE_NAME,
        exempt=exempt,
    )


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Donor support per clipped column.

    Levelled columns carry no bounds. The housing and transport totals keep
    theirs although the stage writes them back around levelled components
    (microcosm#1113): the recomposition holds each to the donor's maximum of
    it (microcosm#1164), and the terminal support check holds the emitted
    totals to the diary's own range, which a stage-time receipt check cannot
    do.
    """

    ranges: dict[str, tuple[float, float]] = {}
    for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS:
        if column in UK_LCFS_RAKED_COLUMNS:
            continue
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


def _encode_consumption_predictors(
    donor: pd.DataFrame, recipient: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    categorical = ("region", "tenure_type", "accommodation_type")
    base_predictors = tuple(
        p for p in UK_LCFS_CONSUMPTION_PREDICTORS if p not in categorical
    )
    donor_work = donor.copy()
    recipient_work = recipient.copy()
    combined = pd.concat(
        [
            donor_work.loc[:, categorical].reset_index(drop=True),
            recipient_work.loc[:, categorical].reset_index(drop=True),
        ],
        ignore_index=True,
    )
    dummies = pd.get_dummies(
        combined.astype(str), columns=list(categorical), dtype=float
    )
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)

    def encode(table: pd.DataFrame, block: pd.DataFrame) -> pd.DataFrame:
        encoded = table.drop(columns=list(categorical), errors="ignore").copy()
        for column in encoded.columns:
            if column == "household_weight":
                continue
            encoded[column] = pd.to_numeric(encoded[column], errors="coerce").fillna(
                0.0
            )
        block = block.copy()
        block.index = encoded.index
        return pd.concat([encoded, block], axis=1)

    donor_encoded = encode(donor_work, dummies.iloc[: len(donor_work)])
    recipient_encoded = encode(recipient_work, dummies.iloc[len(donor_work) :])
    return donor_encoded, recipient_encoded, (*base_predictors, *tuple(dummies.columns))


def _lowercase(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).lower() for column in result.columns]
    return result


#: Columns that mark a frame row as a copy of a source household: the SPI
#: support channel's clones and the high-value home split.
_SOURCE_COPY_COLUMNS = (
    "household_support_clone_index",
    "household_is_high_value_support_copy",
    "high_value_support_copy_index",
)


def _source_households(household: pd.DataFrame) -> np.ndarray | None:
    """Each frame row's source household (``None`` on a frame without copies).

    A frame stacks copies of one source household (the SPI support channel,
    the high-value home split), so a receipt's floor counts these, not rows.
    A frame that marks copies but names no source household is refused, so a
    floor never falls back to counting rows (microcosm#1306 review).
    """

    if "source_household_id" in household.columns:
        return household["source_household_id"].to_numpy()
    marked = [column for column in _SOURCE_COPY_COLUMNS if column in household]
    if marked:
        raise ValueError(
            f"The frame marks copies of source households ({', '.join(marked)}) "
            "but carries no source_household_id, so a receipt cannot count "
            "its distinct source households."
        )
    return None


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0)


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f"LCFS donor is missing required column(s): {missing}.")


def _artifact(stage: SourceStageSpec, role: str) -> Mapping[str, Any]:
    for artifact in stage.artifacts:
        if artifact.get("role") == role:
            return artifact
    raise ValueError(f"{stage.stage} declares no {role!r} artifact.")


def _operation_parameters(
    stage: SourceStageSpec, kind: str, **match: object
) -> Mapping[str, Any]:
    for operation in stage.operations:
        if operation.kind == kind and all(
            operation.parameters.get(key) == value for key, value in match.items()
        ):
            return dict(operation.parameters)
    raise ValueError(f"{stage.stage} declares no {kind!r} operation for {match}.")


def _operation_seed(stage: SourceStageSpec, kind: str) -> int:
    for operation in stage.operations:
        if operation.kind == kind and isinstance(operation.parameters.get("seed"), int):
            return int(operation.parameters["seed"])
    return 0


def _qrf_n_estimators(stage: SourceStageSpec) -> int:
    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf_chain":
            value = operation.parameters.get("n_estimators", 100)
            if isinstance(value, int) and value > 0:
                return value
    return 100


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("LCFS consumption stage requires caller-supplied donor paths.")
    return Path(path).expanduser().resolve()


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
