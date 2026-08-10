"""Bridge between the existing eval harness and DeepEval.

Converts saved eval results (or live CaseResult objects) into DeepEval
test cases with LLM-as-judge metrics. This does NOT replace the
deterministic scorers in metrics.py — it layers on the free-text quality
scoring they deliberately deferred.

Two ways to use it:

    # 1. Score saved results (no API calls to the agent):
    python -m evals.deepeval_bridge evals/results/some-run.json

    # 2. Inside pytest (runs as part of the test suite):
    pytest evals/test_deepeval.py -v
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


DISPOSITION_LABELS = {
    "provisional_credit": "Grant provisional credit to the customer per Reg E/Z",
    "billing_error_hold": "Place a billing error hold per Reg Z procedures",
    "deny": "Deny the dispute claim based on policy criteria",
    "escalate": "Escalate to a senior analyst or specialist team",
}


@dataclass
class DeepEvalCase:
    test_case: LLMTestCase
    case_id: str
    deterministic_correct: bool


rationale_metric = GEval(
    name="Rationale Quality",
    criteria=(
        "The rationale should: "
        "1) cite specific regulation sections (Reg E or Reg Z) relevant to the dispute type, "
        "2) connect the cited regulation to the facts of this specific dispute, "
        "3) explain WHY the chosen disposition follows from the regulation and facts, "
        "4) not contain reasoning that contradicts the chosen disposition."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.6,
)


tool_sequence_metric = GEval(
    name="Tool Usage Quality",
    criteria=(
        "Evaluate the agent's tool usage sequence. A good sequence: "
        "1) retrieves the dispute details first (get_dispute), "
        "2) searches for relevant policies (search_policy), "
        "3) checks transaction history if needed (get_transaction), "
        "4) ends by proposing a resolution (propose_resolution). "
        "Penalise redundant searches, skipped retrieval, and proposals without prior search."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.5,
)


def build_test_case(
    case: dict[str, Any],
    result: dict[str, Any],
) -> DeepEvalCase | None:
    """Convert one case + result pair into a DeepEval test case."""
    disposition = result.get("disposition")
    if disposition is None:
        return None

    expected_desc = DISPOSITION_LABELS.get(
        case["expected_disposition"], case["expected_disposition"]
    )

    cited = result.get("cited", [])
    gold = result.get("gold", case.get("gold_policies", []))

    actual_output = (
        f"Disposition: {disposition}\n"
        f"Cited policies: {', '.join(cited) if cited else 'none'}\n"
        f"Gold policies expected: {', '.join(gold)}\n"
        f"Tool calls made: {result.get('tool_calls', '?')}\n"
        f"Steps taken: {result.get('steps', '?')}"
    )

    expected_output = (
        f"Disposition: {case['expected_disposition']}\n"
        f"Rationale: {expected_desc}\n"
        f"Should cite: {', '.join(gold)}"
    )

    retrieval_context = [
        f"Policy {pid} is the governing policy for this dispute type"
        for pid in gold
    ]

    tc = LLMTestCase(
        input=f"Triage dispute {case['id']}: {case.get('tests', '')}",
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context if retrieval_context else None,
    )

    return DeepEvalCase(
        test_case=tc,
        case_id=case["id"],
        deterministic_correct=result.get("correct", False),
    )


def load_results_file(path: Path) -> tuple[list[dict], list[dict]]:
    """Load a saved results JSON and return (cases_meta, case_results)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    case_results = data.get("cases", [])

    from evals.metrics import load_cases
    all_cases = {c["id"]: c for c in load_cases()}

    cases_meta = []
    for r in case_results:
        cid = r.get("case_id", "")
        if cid in all_cases:
            cases_meta.append(all_cases[cid])
        else:
            cases_meta.append({"id": cid, "expected_disposition": r.get("expected", ""), "gold_policies": r.get("gold", [])})

    return cases_meta, case_results


def main() -> None:
    """Score a saved results file with DeepEval metrics."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m evals.deepeval_bridge <results.json>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    cases_meta, case_results = load_results_file(path)

    from deepeval import evaluate

    test_cases = []
    for case_meta, result in zip(cases_meta, case_results):
        dc = build_test_case(case_meta, result)
        if dc:
            test_cases.append(dc.test_case)
            print(f"  {dc.case_id}: {'PASS' if dc.deterministic_correct else 'FAIL'} (deterministic)")

    if not test_cases:
        print("No scoreable cases found.")
        sys.exit(1)

    print(f"\nScoring {len(test_cases)} cases with DeepEval LLM-as-judge...\n")
    evaluate(test_cases, [rationale_metric, tool_sequence_metric])


if __name__ == "__main__":
    main()
