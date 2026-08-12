"""FX Rates MCP server — the second server in the multi-server demo.

Exposes two read-only tools:

    get_fx_rate     mid-market rate for a currency pair on a date
    convert_currency  convert an amount at the historical rate

No writes, no auth, no database. Purely a data lookup. In production this
would proxy a market data feed (Bloomberg, Refinitiv, ECB) behind the same
MCP interface, so the agent code would not change.

Run standalone:
    mcp dev fx_server/app.py

Run as a subprocess (the agent loop does this automatically):
    python -m fx_server.app
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from fx_server.rates import convert, get_rate

mcp = MCPServer(
    name="fx-rates",
    instructions=(
        "Foreign exchange rate lookup for Northwind Bank. "
        "Use these tools when a dispute involves a cross-currency transaction "
        "to verify the converted amount on the customer's statement."
    ),
)


@mcp.tool()
def get_fx_rate(base: str, quote: str, date: str) -> dict:
    """Look up the mid-market exchange rate for a currency pair on a given date.

    Args:
        base: The base currency code (e.g. 'EUR').
        quote: The quote currency code (e.g. 'USD').
        date: The date to look up, ISO format (e.g. '2026-07-11').

    Returns the rate such that 1 unit of base = rate units of quote.
    For example, EUR/USD = 1.0985 means 1 EUR = 1.0985 USD.
    """
    return get_rate(base, quote, date)


@mcp.tool()
def convert_currency(
    amount: float, from_currency: str, to_currency: str, date: str
) -> dict:
    """Convert an amount between currencies at the historical mid-market rate.

    Use this to verify whether a cross-currency charge on a customer's
    statement matches the expected conversion. A significant discrepancy
    (more than ~2% from mid-market) may indicate a markup worth noting.

    Args:
        amount: The amount to convert.
        from_currency: Source currency code (e.g. 'EUR').
        to_currency: Target currency code (e.g. 'USD').
        date: The transaction date, ISO format.
    """
    return convert(amount, from_currency, to_currency, date)


if __name__ == "__main__":
    mcp.run()
