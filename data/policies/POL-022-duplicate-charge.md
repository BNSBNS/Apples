# POL-022 — Merchant Dispute: Duplicate or Repeated Charge

*Northwind Bank (fictional institution — synthetic policy for demo purposes).*
Category: merchant error.

## Clause 22.1 — Definition

A duplicate charge is **two or more settled transactions** with the same merchant, the same amount,
and the same card, within a **72-hour** window, where the customer intended only one purchase.

## Clause 22.2 — Not a duplicate

Recurring subscriptions billed on their normal cycle, split-tender transactions, partial captures
of a single authorisation, and separate purchases that coincidentally match in amount are **not**
duplicates. An authorisation hold sitting alongside its own settled transaction is not a duplicate
and typically clears within 3 business days.

## Clause 22.3 — Resolution

Where Clause 22.1 is satisfied and the merchant has not already refunded, credit the lesser
duplicate amount. No merchant-contact precondition applies — duplicates are resolved directly by
the bank.

## Clause 22.4 — Filing deadline

Within **90 calendar days** of the later transaction date.

> **Distinguish from POL-023.** A subscription the customer believes they cancelled is a
> **cancelled recurring billing** matter under POL-023, not a duplicate — even where the amounts
> match exactly across cycles.
