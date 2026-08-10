"""DeepEval tests for the dispute triage agent.

Run with:  pytest evals/test_deepeval.py -v
           deepeval test run evals/test_deepeval.py

These tests score agent OUTPUT quality with LLM-as-judge — they do NOT
replace the deterministic metrics in metrics.py. Think of this as the
"rationale quality" layer that metrics.py explicitly deferred.

Requires: pip install -e ".[eval]"
          OPENAI_API_KEY set (DeepEval uses OpenAI as the default judge)

To run against saved results without re-running the agent:
    python -m evals.deepeval_bridge evals/results/<run>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase
except ImportError:
    pytest.skip("deepeval not installed — run: pip install -e '.[eval]'", allow_module_level=True)

from evals.deepeval_bridge import (
    build_test_case,
    rationale_metric,
    tool_sequence_metric,
)
from evals.metrics import load_cases

RESULTS_DIR = Path(__file__).parent / "results"


def _find_latest_results() -> Path | None:
    """Find the most recent results file (excluding pre-metrics-v1)."""
    candidates = [
        p for p in RESULTS_DIR.glob("*.json")
        if "pre-metrics" not in str(p)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_test_cases() -> list[tuple[str, LLMTestCase]]:
    """Load test cases from the latest results file."""
    results_path = _find_latest_results()
    if results_path is None:
        return []

    data = json.loads(results_path.read_text(encoding="utf-8"))
    case_results = data.get("cases", [])
    all_cases = {c["id"]: c for c in load_cases()}

    pairs = []
    for r in case_results:
        cid = r.get("case_id", "")
        case_meta = all_cases.get(cid)
        if not case_meta:
            continue
        dc = build_test_case(case_meta, r)
        if dc:
            pairs.append((dc.case_id, dc.test_case))
    return pairs


_cached_cases = _load_test_cases()

skip_no_results = pytest.mark.skipif(
    not _cached_cases,
    reason=(
        "No eval results found. Run the eval suite first:\n"
        "  python -m evals.run_evals --provider local --label test-run\n"
        "Then re-run these tests."
    ),
)


@skip_no_results
@pytest.mark.parametrize(
    "case_id,test_case",
    _cached_cases,
    ids=[c[0] for c in _cached_cases],
)
def test_rationale_quality(case_id: str, test_case: LLMTestCase):
    """LLM judge: does the rationale cite regulations and connect them to facts?"""
    assert_test(test_case, [rationale_metric])


@skip_no_results
@pytest.mark.parametrize(
    "case_id,test_case",
    _cached_cases,
    ids=[c[0] for c in _cached_cases],
)
def test_tool_usage_quality(case_id: str, test_case: LLMTestCase):
    """LLM judge: did the agent use tools in a sensible order?"""
    assert_test(test_case, [tool_sequence_metric])


# --- Smoke test that works without saved results ---

SMOKE_CASE = {
    "id": "SMOKE-001",
    "expected_disposition": "provisional_credit",
    "gold_policies": ["POL-001"],
    "tests": "smoke test: unauthorized debit, timely filing",
}

SMOKE_RESULT = {
    "case_id": "SMOKE-001",
    "disposition": "provisional_credit",
    "correct": True,
    "cited": ["POL-001"],
    "gold": ["POL-001"],
    "tool_calls": 4,
    "steps": 3,
}


def test_smoke_deepeval_integration():
    """Verify DeepEval is installed and metrics can be instantiated."""
    dc = build_test_case(SMOKE_CASE, SMOKE_RESULT)
    assert dc is not None
    assert dc.case_id == "SMOKE-001"
    assert dc.deterministic_correct is True
    assert dc.test_case.input is not None
    assert dc.test_case.actual_output is not None
    assert dc.test_case.expected_output is not None
