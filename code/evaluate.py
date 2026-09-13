"""
Evaluation & verification suite for Buy or Wait? AI financial decision agent.

Asserts contract compliance, format validity, static determinism boundary checks,
and evaluates prediction alignment against dataset/sample_requests.csv.
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

EXPECTED_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation"
]

VALID_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
VALID_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def test_determinism_boundary() -> None:
    """Asserts statically that financial_engine.py and decision_engine.py do not import LLM clients."""
    target_files = [config.REPO_ROOT / "code" / "financial_engine.py", config.REPO_ROOT / "code" / "decision_engine.py"]
    for filepath in target_files:
        with open(filepath, mode="r", encoding="utf-8") as f:
            code = f.read()
            for forbidden in ["openai", "langchain_openai", "anthropic", "google.generativeai", "google.genai"]:
                assert forbidden not in code, f"Determinism violation: {forbidden} found in {filepath.name}"
    print("PASS: Determinism boundary static import check passed.")


def test_output_csv_format() -> None:
    """Validates structure, row count, enums, and numerical bounds on dataset/output.csv."""
    assert config.OUTPUT_CSV_PATH.exists(), f"output.csv missing at {config.OUTPUT_CSV_PATH}"
    
    with open(config.OUTPUT_CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert list(reader.fieldnames or []) == EXPECTED_COLUMNS, f"Columns mismatch in {config.OUTPUT_CSV_PATH}"
        rows = list(reader)

    with open(config.REQUESTS_CSV, mode="r", encoding="utf-8") as f:
        requests_rows = list(csv.DictReader(f))

    assert len(rows) == len(requests_rows), f"Row count mismatch: output.csv has {len(rows)}, requests.csv has {len(requests_rows)}"

    for idx, row in enumerate(rows):
        req = requests_rows[idx]
        assert row["request_id"] == req["request_id"], f"Row {idx} request_id mismatch"
        
        amt_safe = float(row["amount_safe_to_pay"])
        req_amt = float(req["requested_amount"])
        assert 0.0 <= amt_safe <= req_amt + 0.01, f"Row {idx} amount_safe_to_pay {amt_safe} out of bounds [0, {req_amt}]"
        
        assert row["affordability_status"] in VALID_STATUSES, f"Row {idx} invalid status {row['affordability_status']}"
        assert row["recommended_payment_method"] in VALID_METHODS, f"Row {idx} invalid method {row['recommended_payment_method']}"

    print(f"PASS: output.csv verified ({len(rows)} rows, exact column parity, valid enums and bounds).")


def test_sample_requests_benchmark() -> None:
    """Evaluates pipeline against dataset/sample_requests.csv benchmark."""
    from data_loader import get_data_loader
    from graph_builder import get_compiled_graph

    loader = get_data_loader()
    graph = get_compiled_graph()

    with open(config.SAMPLE_REQUESTS_CSV, mode="r", encoding="utf-8") as f:
        samples = list(csv.DictReader(f))

    matches_status = 0
    matches_method = 0

    for sample in samples:
        r_id = sample["request_id"]
        req_row = next((r for r in loader.requests if r["request_id"] == r_id), None)
        if not req_row:
            req_row = {
                "request_id": sample["request_id"],
                "user_id": sample["user_id"],
                "request_date": sample["request_date"],
                "request_type": sample["request_type"],
                "requested_amount": float(sample["requested_amount"]),
                "desired_completion_date": sample["desired_completion_date"],
                "allows_partial_payment": str(sample["allows_partial_payment"]).lower() == "true",
                "request_text": sample["request_text"]
            }

        res = graph.invoke({"request": req_row})
        dec = res.get("decision", {})

        if dec.get("affordability_status") == sample.get("affordability_status"):
            matches_status += 1
        if dec.get("recommended_payment_method") == sample.get("recommended_payment_method"):
            matches_method += 1

    print(f"Benchmark on {len(samples)} sample requests:")
    print(f"  - Affordability Status Accuracy: {matches_status}/{len(samples)} ({matches_status/len(samples)*100:.1f}%)")
    print(f"  - Payment Method Accuracy: {matches_method}/{len(samples)} ({matches_method/len(samples)*100:.1f}%)")


def main() -> None:
    print("Running evaluation suite...")
    test_determinism_boundary()
    test_output_csv_format()
    test_sample_requests_benchmark()
    print("All evaluation checks passed successfully.")


if __name__ == "__main__":
    main()
