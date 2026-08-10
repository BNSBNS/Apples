# POL-030 — ATM Disputes: Cash Not Dispensed and Partial Dispense

*Northwind Bank (fictional institution — synthetic policy for demo purposes).*

## Clause 30.1 — Scope

Covers withdrawals where cash was not dispensed, only partially dispensed, or the machine captured
the card mid-transaction. ATM transactions are electronic fund transfers and therefore also fall
under the Regulation E provisional-credit obligations in POL-001.

## Clause 30.2 — Terminal reconciliation

Before any determination, the terminal's electronic journal and end-of-day cash reconciliation must
be obtained. A confirmed **cash overage** at that terminal for the business day is strong evidence
supporting the claim.

## Clause 30.3 — On-us versus foreign terminals

Where the ATM is a Northwind-owned terminal, reconciliation is available within **2 business
days**. For third-party terminals, allow **10 business days** for the acquirer's response — but
this does **not** extend the provisional-credit deadline in POL-001 Clause 1.1.

## Clause 30.4 — Partial dispense

Credit only the difference between the amount debited and the amount actually dispensed, as
established by the journal.

> **Note.** The 10-business-day acquirer window in Clause 30.3 is frequently confused with the
> 10-business-day provisional-credit deadline in POL-001. They are unrelated and run concurrently.
