# POL-032 — Chargeback Representment and Second Presentment

*Northwind Bank (fictional institution — synthetic policy for demo purposes).*

## Clause 32.1 — What representment is

Representment occurs when the merchant's acquirer contests a chargeback and re-presents the
transaction with supporting evidence. It is a **network-level** process between issuer and
acquirer, not a customer-facing dispute stage.

## Clause 32.2 — Issuer response window

Northwind must accept or contest a representment within **30 calendar days** of receipt. Missing
this window forfeits the claim and the provisional credit becomes a loss to the bank.

## Clause 32.3 — Customer notification

Where representment succeeds and provisional credit must be reversed, the customer receives the
**5 business days'** advance notice required by POL-001 Clause 1.4. Representment is not itself a
new denial and does not restart any dispute window.

## Clause 32.4 — Pre-arbitration

Where the customer supplies new evidence that rebuts the representment, the claim may proceed to
pre-arbitration. This requires supervisory approval and is out of scope for first-line handling.

> **Scope note.** First-line triage never initiates or responds to representment. If a claim record
> shows an open representment, disposition is `escalate` under POL-031 Clause 31.2.
