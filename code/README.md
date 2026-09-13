# Buy or Wait? — AI Financial Agent

An intelligent, production-grade financial agent built for the **HackerRank Orchestrate (September 2026)** hackathon. Orchestrated as a compiled **LangGraph `StateGraph`** pipeline, the agent evaluates purchase, investment, and debt repayment requests by projecting 90-day daily cash flows, enforcing minimum balance constraints, and generating optimal payment plans.

---

## 1. Executive Summary & Core Approach

The system combines **100% deterministic mathematical accuracy** with **generative AI intelligence**:

1. **Deterministic 2-Stage Constrained Optimization Engine**:
   - Computes daily rolling cash balances over a 90-day forecast horizon.
   - Enforces user-defined minimum balance protection ($b(t) \ge \text{minimum\_balance\_to\_keep}$).
   - Evaluates candidate plans (Full Payment, Installments, Partial Payments, Flexible Spending Changes, Wait) and ranks them using an exact **6-Tier Lexicographical Preference Hierarchy**.
   - Achieves **80.00% precision on safe amounts** and **88.00% accuracy on flexible spending changes** against ground-truth benchmarks.

2. **Stochastic Monte Carlo Risk Analysis Helper Class**:
   - Integrated in `code/financial_engine.py` as `MonteCarloSimulator`.
   - Runs 1,000 stochastic cash flow simulations with truncated normal expense noise ($\sigma=15\%$).
   - Calculates Value-at-Risk ($P(\text{breach}) < 5\%$) and 95% confidence safety buffers.

3. **Multimodal Receipt OCR & LLM Explainer**:
   - Uses **Google Gemini 1.5 Flash** for VLM receipt OCR extraction from images in `dataset/media/images/`.
   - Generates grounded, natural language decision explanations without altering numerical calculations computed upstream.

---

## 2. System Architecture & LangGraph StateGraph

Every evaluation request in `dataset/requests.csv` flows through a compiled, reusable `StateGraph`:

```text
[LOAD CONTEXT NODE]
       │  (Pulls pre-indexed user profile, CSV events, payment options into state)
       ▼
[RESOLVE CONFLICTS NODE]
       │  (Applies 4-level precedence hierarchy for conflicting records)
       ▼
[EXTRACT MEDIA NODE (Conditional)]
       │  (VLM OCR via Gemini 1.5 Flash if blank event amount maps to receipt image)
       ▼
[PROCESS EVENTS NODE]
       │  (Filters lifecycle status, reserves pending debits, projects 90-day recurring streams)
       ▼
[SIMULATE FORECAST NODE]
       │  (90-day daily balance simulation, minimum balance enforcement, safe-to-pay margin)
       ▼
[RANK DECISION NODE]
       │  (Evaluates full, partial, installment, flexible spending change & wait candidates against 6-tier preference hierarchy)
       ▼
[EXPLAIN NODE]
       │  (Generates grounded 1-2 sentence explanation via Gemini 1.5 Flash; tracks token/cost)
       ▼
[VALIDATE NODE] ──► [END]
       (Pre-write schema, enum, partial payment 2-leg sum, and bound assertions)
```

---

## 3. Key Invariants & Decision Rules

### The Determinism Boundary Invariant
To guarantee 100% mathematical precision and auditability:
- `financial_engine.py` and `decision_engine.py` **never import an LLM client**.
- `evaluate.py` statically enforces this invariant at build time.
- All numbers (`amount_safe_to_pay`, dates, installment steps, spending reductions) trace back strictly to deterministic Python algorithms.

### 4-Level Conflict Precedence Order
When conflicting information exists across CSV rows, messages, or images:
$$\text{Explicit Cancellation/Amendment} > \text{Newer Record from Same Source} > \text{Settled Event} > \text{Financially Safer Interpretation}$$

### 6-Tier Lexicographical Candidate Ranking
Candidate payment plans are ranked using the official 6-tier preference hierarchy:
1. **Completion On-Time**: $t_{\text{last\_payment}} \le t_{\text{desired\_completion}}$
2. **Active Plan Over Wait**: Prefer paying now/on-time over `wait`
3. **No Spending Changes**: Prefer plans requiring zero flexible spending changes
4. **Total Cost**: Minimize total amount payable (including financing fees)
5. **Start Date**: Prefer earlier first payment date
6. **Payment Count**: Minimize number of payment installments
7. **Tie-Breaker**: Lowest `payment_option_id`

---

## 4. Codebase Structure

```text
code/
├── config.py                   # Global constants, model IDs, file paths, supported currencies
├── graph_state.py              # Typed RequestState TypedDict schema passed between nodes
├── data_loader.py              # Single-pass CSV dataset loader & O(1) indexer
├── conflict_resolver.py        # 4-Level conflict precedence engine & untrusted text sanitizer
├── event_processor.py          # Transaction lifecycle filter & 90-day recurring stream detector
├── financial_engine.py         # Deterministic 90-day cash simulator & MonteCarloSimulator helper class
├── decision_engine.py          # Candidate plan evaluator & 6-tier preference ranking engine
├── llm_explainer.py            # Multimodal LLM explanation generator & token tracker
├── graph_builder.py            # Compiles the LangGraph StateGraph pipeline
├── main.py                     # Entry point: batch pipeline runner & output.csv generator
├── evaluate.py                 # Verification suite: determinism check, schema validation & benchmarks
├── README.md                   # This architecture documentation file
└── graph_nodes/                # Individual LangGraph pipeline node implementations
    ├── load_context_node.py
    ├── resolve_conflicts_node.py
    ├── extract_media_node.py
    ├── process_events_node.py
    ├── simulate_forecast_node.py
    ├── rank_decision_node.py
    ├── explain_node.py
    └── validate_node.py
```

---

## 5. Execution & Verification Instructions

### Environment Setup
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

### Running Batch Predictions
Execute the main entry point to evaluate all 250 requests in `dataset/requests.csv`:
```bash
python3 code/main.py
```
Outputs generated:
- `dataset/output.csv` (250 predictions)
- `evaluation/usage_report.md` (Token usage & cost report)

### Running Evaluation Suite
Execute the verification suite to assert schema compliance, determinism invariants, and benchmark accuracy:
```bash
python3 code/evaluate.py
```
