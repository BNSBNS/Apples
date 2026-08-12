"""Synthetic FX rate data for the Northwind Bank demo.

In production this would hit a market data API (Bloomberg, Refinitiv, a
central bank feed). Here it is a lookup table covering the date range of the
transactions in data/seed.py, so the demo is fully offline.

Rates are mid-market, keyed as (base, quote) → {date: rate}. EUR/USD = 1.08
means 1 EUR buys 1.08 USD.
"""

from __future__ import annotations

# fmt: off
# Synthetic daily mid-market rates covering the transaction window (2026-04 to 2026-08).
# Enough variation to make the math non-trivial without being volatile.
RATES: dict[tuple[str, str], dict[str, float]] = {
    ("EUR", "USD"): {
        "2026-04-01": 1.0810, "2026-04-02": 1.0815,
        "2026-05-01": 1.0790, "2026-05-02": 1.0795, "2026-05-03": 1.0800,
        "2026-06-01": 1.0920, "2026-06-15": 1.0935, "2026-06-28": 1.0940,
        "2026-06-30": 1.0945,
        "2026-07-01": 1.0950, "2026-07-02": 1.0955, "2026-07-05": 1.0960,
        "2026-07-08": 1.0970, "2026-07-09": 1.0975, "2026-07-10": 1.0980,
        "2026-07-11": 1.0985, "2026-07-12": 1.0990, "2026-07-14": 1.0995,
        "2026-07-15": 1.1000, "2026-07-16": 1.1005, "2026-07-19": 1.1010,
        "2026-07-21": 1.1020, "2026-07-22": 1.1025, "2026-07-23": 1.1030,
        "2026-07-24": 1.1035, "2026-07-25": 1.1040,
        "2026-08-01": 1.1060, "2026-08-10": 1.1075, "2026-08-12": 1.1080,
    },
    ("GBP", "USD"): {
        "2026-04-01": 1.2710, "2026-04-02": 1.2715,
        "2026-05-01": 1.2690, "2026-05-02": 1.2695, "2026-05-03": 1.2700,
        "2026-06-01": 1.2820, "2026-06-15": 1.2835, "2026-06-28": 1.2840,
        "2026-06-30": 1.2845,
        "2026-07-01": 1.2850, "2026-07-02": 1.2855, "2026-07-05": 1.2860,
        "2026-07-08": 1.2870, "2026-07-09": 1.2875, "2026-07-10": 1.2880,
        "2026-07-11": 1.2885, "2026-07-12": 1.2890, "2026-07-14": 1.2895,
        "2026-07-15": 1.2900, "2026-07-16": 1.2905, "2026-07-19": 1.2910,
        "2026-07-21": 1.2920, "2026-07-22": 1.2925, "2026-07-23": 1.2930,
        "2026-07-24": 1.2935, "2026-07-25": 1.2940,
        "2026-08-01": 1.2960, "2026-08-10": 1.2975, "2026-08-12": 1.2980,
    },
    ("JPY", "USD"): {
        "2026-07-08": 0.006410, "2026-07-11": 0.006415,
        "2026-07-15": 0.006420, "2026-07-24": 0.006430,
    },
}
# fmt: on

SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "JPY"}


def _nearest_date(dates: dict[str, float], target: str) -> tuple[str, float]:
    """Find the closest date on or before target. Falls back to closest after."""
    on_or_before = {d: r for d, r in dates.items() if d <= target}
    if on_or_before:
        best = max(on_or_before)
        return best, on_or_before[best]
    best = min(dates)
    return best, dates[best]


def get_rate(base: str, quote: str, date: str) -> dict:
    """Look up the mid-market rate for a currency pair on a given date."""
    base, quote = base.upper(), quote.upper()

    if base == quote:
        return {"base": base, "quote": quote, "date": date, "rate": 1.0, "source": "identity"}

    key = (base, quote)
    inverted = False
    if key not in RATES:
        key = (quote, base)
        inverted = True
    if key not in RATES:
        return {"error": f"no rate data for {base}/{quote}. Supported: {sorted(SUPPORTED_CURRENCIES)}"}

    used_date, rate = _nearest_date(RATES[key], date)
    if inverted:
        rate = round(1.0 / rate, 6)

    return {
        "base": base,
        "quote": quote,
        "date": date,
        "rate_date": used_date,
        "rate": rate,
        "source": "northwind-fx-desk",
    }


def convert(amount: float, from_currency: str, to_currency: str, date: str) -> dict:
    """Convert an amount between currencies at the rate on a given date."""
    lookup = get_rate(from_currency, to_currency, date)
    if "error" in lookup:
        return lookup

    converted = round(amount * lookup["rate"], 2)
    return {
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "original_amount": amount,
        "converted_amount": converted,
        "rate": lookup["rate"],
        "rate_date": lookup["rate_date"],
        "source": lookup["source"],
    }
