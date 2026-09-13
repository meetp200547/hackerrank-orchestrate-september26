# Token Usage and Cost Analysis — Buy or Wait? AI Agent

This report summarizes token usage, model calls, and estimated costs for the full dataset run on `dataset/requests.csv` (250 requests).

---

## Executive Summary

- **Total Evaluation Requests**: 250
- **Total Model Calls**: 261
- **Total Input Tokens**: 7,436
- **Total Output Tokens**: 3,018
- **Total Tokens Combined**: 10,454
- **Average Tokens / Request**: 41.82
- **Estimated Total Cost**: $0.0015 USD
- **Estimated Cost / Request**: $0.0000 USD

---

## Per-Model Breakout

| Provider / Model | Calls | Input Tokens | Output Tokens | Total Tokens | Estimated Cost (USD) |
|---|---|---|---|---|---|
| deterministic / template-generator | 210 | 0 | 0 | 0 | $0.0000 |
| google / gemini-3.5-flash-lite | 11 | 1,650 | 275 | 1,925 | $0.0002 |
| google-genai / gemini-3.5-flash-lite | 40 | 5,786 | 2,743 | 8,529 | $0.0013 |

---

## Notes & Compliance

- Deterministic core modules (`financial_engine.py`, `decision_engine.py`) perform zero LLM calls.
- Model calls are strictly isolated to VLM/OCR image extraction (`extract_media_node`) and grounded explanation generation (`explain_node`).
