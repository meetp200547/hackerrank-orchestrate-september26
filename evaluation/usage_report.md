# Token Usage and Cost Analysis — Buy or Wait? AI Agent

This report summarizes token usage, model calls, and estimated costs for the full dataset run on `dataset/requests.csv` (250 requests).

---

## Executive Summary

- **Total Evaluation Requests**: 250
- **Total Model Calls**: 261
- **Total Input Tokens**: 11,636
- **Total Output Tokens**: 4,860
- **Total Tokens Combined**: 16,496
- **Average Tokens / Request**: 65.98
- **Estimated Total Cost**: $0.0023 USD
- **Estimated Cost / Request**: $0.0000 USD

---

## Per-Model Breakout

| Provider / Model | Calls | Input Tokens | Output Tokens | Total Tokens | Estimated Cost (USD) |
|---|---|---|---|---|---|
| google-genai / gemini-3.5-flash-lite | 69 | 9,986 | 4,585 | 14,571 | $0.0021 |
| google / gemini-3.5-flash-lite | 11 | 1,650 | 275 | 1,925 | $0.0002 |
| deterministic / template-generator | 181 | 0 | 0 | 0 | $0.0000 |

---

## Notes & Compliance

- Deterministic core modules (`financial_engine.py`, `decision_engine.py`) perform zero LLM calls.
- Model calls are strictly isolated to VLM/OCR image extraction (`extract_media_node`) and grounded explanation generation (`explain_node`).
