# POL-031 — Escalation to Fraud Operations

*Northwind Bank (fictional institution — synthetic policy for demo purposes).*

## Clause 31.1 — Mandatory escalation

Escalate to Fraud Operations, and do **not** dispose of the claim at first line, where any of the
following holds:

- Disputed amount exceeds **USD 5,000** in aggregate for the claim.
- The customer has filed **three or more** claims in the preceding **12 months** (see POL-003
  Clause 3.3).
- The claim involves a wire transfer, ACH origination, or any transfer to an external account.
- Account takeover is suspected: credential change, contact-detail change, or device enrolment
  within **30 days** before the disputed transaction.
- The customer reports coercion, a third party known to them, or a romance or investment scam.

## Clause 31.2 — Discretionary escalation

Escalate where the evidence is genuinely balanced under POL-012, where the customer disputes a
prior denial, or where the merchant is subject to an active investigation.

## Clause 31.3 — Escalation does not pause obligations

Escalation does **not** extend the provisional-credit deadline in POL-001 Clause 1.1 unless a
withholding ground in POL-003 independently applies.

## Clause 31.4 — Handling

An escalated claim is recorded with disposition `escalate`, a written rationale citing the specific
clause triggered, and all evidence gathered to that point.
