"""
Main entry point for Buy or Wait? AI financial agent.

Runs the compiled LangGraph StateGraph over all requests in dataset/requests.csv,
generates dataset/output.csv, and logs token usage and cost to evaluation/usage_report.md.
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path
from typing import Any

# Ensure code/ is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from data_loader import get_data_loader
from graph_builder import get_compiled_graph


def write_output_csv(output_rows: list[dict[str, Any]], target_file: Path) -> None:
    headers = [
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation"
    ]
    with open(target_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({
                "request_id": row["request_id"],
                "amount_safe_to_pay": row["amount_safe_to_pay"],
                "affordability_status": row["affordability_status"],
                "recommended_payment_method": row["recommended_payment_method"],
                "payment_plan": row["payment_plan"],
                "earliest_date_for_full_payment": row.get("earliest_date_for_full_payment") or "",
                "spending_changes_needed": row["spending_changes_needed"],
                "decision_explanation": row["decision_explanation"]
            })


def generate_usage_report(token_entries: list[dict[str, Any]], total_requests: int, report_file: Path) -> None:
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Model aggregation
    by_model: dict[str, dict[str, Any]] = {}
    total_calls = len(token_entries)
    total_in_tokens = 0
    total_out_tokens = 0
    total_cost = 0.0

    for entry in token_entries:
        model = entry.get("model", "unknown")
        provider = entry.get("provider", "unknown")
        key = f"{provider} / {model}"
        
        if key not in by_model:
            by_model[key] = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "cost": 0.0}
            
        in_t = entry.get("input_tokens", 0)
        out_t = entry.get("output_tokens", 0)
        c = entry.get("estimated_cost_usd", 0.0)

        by_model[key]["calls"] += 1
        by_model[key]["in_tokens"] += in_t
        by_model[key]["out_tokens"] += out_t
        by_model[key]["cost"] += c

        total_in_tokens += in_t
        total_out_tokens += out_t
        total_cost += c

    total_tokens = total_in_tokens + total_out_tokens
    avg_tokens_per_req = total_tokens / total_requests if total_requests > 0 else 0.0
    avg_cost_per_req = total_cost / total_requests if total_requests > 0 else 0.0

    content = f"""# Token Usage and Cost Analysis — Buy or Wait? AI Agent

This report summarizes token usage, model calls, and estimated costs for the full dataset run on `dataset/requests.csv` ({total_requests} requests).

---

## Executive Summary

- **Total Evaluation Requests**: {total_requests}
- **Total Model Calls**: {total_calls}
- **Total Input Tokens**: {total_in_tokens:,}
- **Total Output Tokens**: {total_out_tokens:,}
- **Total Tokens Combined**: {total_tokens:,}
- **Average Tokens / Request**: {avg_tokens_per_req:.2f}
- **Estimated Total Cost**: ${total_cost:.4f} USD
- **Estimated Cost / Request**: ${avg_cost_per_req:.4f} USD

---

## Per-Model Breakout

| Provider / Model | Calls | Input Tokens | Output Tokens | Total Tokens | Estimated Cost (USD) |
|---|---|---|---|---|---|
"""
    for key, data in by_model.items():
        t_tok = data["in_tokens"] + data["out_tokens"]
        content += f"| {key} | {data['calls']} | {data['in_tokens']:,} | {data['out_tokens']:,} | {t_tok:,} | ${data['cost']:.4f} |\n"

    content += """
---

## Notes & Compliance

- Deterministic core modules (`financial_engine.py`, `decision_engine.py`) perform zero LLM calls.
- Model calls are strictly isolated to VLM/OCR image extraction (`extract_media_node`) and grounded explanation generation (`explain_node`).
"""

    with open(report_file, mode="w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    print("Initializing Buy or Wait? AI Financial Agent...")
    loader = get_data_loader()
    graph = get_compiled_graph()
    
    requests = loader.requests
    print(f"Loaded {len(requests)} requests from dataset/requests.csv")

    output_rows: list[dict[str, Any]] = []
    all_token_entries: list[dict[str, Any]] = []

    print("Evaluating requests via LangGraph StateGraph pipeline...")
    for idx, req in enumerate(requests, 1):
        initial_state = {"request": req}
        result = graph.invoke(initial_state)

        dec = result.get("decision", {})
        output_row = {
            "request_id": req["request_id"],
            "amount_safe_to_pay": dec.get("amount_safe_to_pay", 0.0),
            "affordability_status": dec.get("affordability_status", "not_affordable"),
            "recommended_payment_method": dec.get("recommended_payment_method", "not_recommended"),
            "payment_plan": dec.get("payment_plan", "none"),
            "earliest_date_for_full_payment": dec.get("earliest_date_for_full_payment") or "",
            "spending_changes_needed": dec.get("spending_changes_needed", "none"),
            "decision_explanation": dec.get("decision_explanation", "")
        }
        output_rows.append(output_row)
        
        token_entries = result.get("token_usage", [])
        all_token_entries.extend(token_entries)

        if idx % 50 == 0 or idx == len(requests):
            print(f"Processed [{idx}/{len(requests)}] requests.")

    # Write dataset/output.csv
    write_output_csv(output_rows, config.OUTPUT_CSV_PATH)
    print(f"Saved output predictions to {config.OUTPUT_CSV_PATH}")

    # Generate evaluation/usage_report.md
    generate_usage_report(all_token_entries, len(requests), config.USAGE_REPORT_PATH)
    print(f"Saved token usage report to {config.USAGE_REPORT_PATH}")


if __name__ == "__main__":
    main()
