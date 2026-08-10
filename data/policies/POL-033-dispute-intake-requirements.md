# POL-033 — Dispute Intake: Minimum Required Information

*Northwind Bank (fictional institution — synthetic policy for demo purposes).*

## Clause 33.1 — Mandatory fields

No claim may be dispositioned without: the customer identifier; the specific transaction
identifier(s); the disputed amount; the dispute category (unauthorised / not received / not as
described / duplicate / cancelled recurring / ATM); and the date the customer became aware.

## Clause 33.2 — Insufficient information

Where a mandatory field cannot be established from the record, the correct disposition is
`escalate` with a note stating the missing field. **Guessing a category is a policy violation** —
category determines which timeline and evidence standard applies.

## Clause 33.3 — Transaction must exist

A dispute referencing a transaction that cannot be located on the account must not be denied on the
merits. It is recorded as `escalate` for manual identification — the transaction may be pending,
posted to a linked account, or referenced by a merchant descriptor differing from the trading name.

## Clause 33.4 — Customer attestation

For unauthorised-transaction claims the customer must attest that they did not authorise the
transaction and did not permit another person to use the access device. Absent attestation, the
claim is a merchant dispute (POL-020 to POL-023), not a fraud claim.
