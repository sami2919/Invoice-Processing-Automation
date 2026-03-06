# Invoice Processing AI — Multi-Agent System for Automated AP Workflow

A production-ready multi-agent pipeline built with **LangGraph + Grok** that automates the full accounts-payable workflow: invoice ingestion → extraction → validation → fraud detection → approval → payment → audit trail. Includes a Streamlit HITL dashboard, CLI, batch mode, and 180 automated tests.

---

## Business Context

A PE-backed manufacturer was losing over $2M/year to manual invoice processing — a 5-day cycle time, a 30% error rate on line-item validation, and no systematic fraud detection. Duplicate invoices alone accounted for an estimated 2% in overpayments. This system replaces the manual workflow with a multi-agent AI pipeline that extracts structured data from any invoice format, runs deterministic business-rule checks, scores fraud risk across 14 signals, and routes borderline cases to a human reviewer in a purpose-built Streamlit interface — all in under 30 seconds per invoice.

---

## Architecture

```mermaid
graph TD
    A[Invoice Input<br/>TXT · JSON · CSV · XML · PDF] --> B[Extraction Agent<br/>Grok structured output<br/>OCR correction · fuzzy match]
    B --> C[Validation Agent<br/>14 deterministic checks<br/>stock · math · duplicates]
    C -->|Valid| D[Fraud Detection<br/>14 weighted signals<br/>score 0-100]
    C -->|Invalid + retries left| B
    C -->|Invalid + retries exhausted| G[Rejection Node]
    D --> E[Approval Agent<br/>6-step routing]
    E -->|Auto-approve<br/>amount < $10K · risk < 30| F[Payment Agent<br/>mock_payment API]
    E -->|Human review required| H[HITL Review<br/>LangGraph interrupt]
    E -->|High risk >= 70| G
    H -->|Approved| F
    H -->|Rejected| G
    F --> I[Explanation Agent<br/>VP-readable summary]
    G --> I
    I --> J[END<br/>Full audit trail]

    style A fill:#e8f4f8,stroke:#2196F3
    style D fill:#fff3e0,stroke:#FF9800
    style E fill:#f3e5f5,stroke:#9C27B0
    style H fill:#fce4ec,stroke:#E91E63
    style F fill:#e8f5e9,stroke:#4CAF50
    style G fill:#ffebee,stroke:#F44336
```

![Pipeline Diagram](pipeline_diagram.png)

The pipeline follows a directed workflow with self-correction loops:

1. **Extraction Agent** — Grok structured output with OCR correction and fuzzy item matching
2. **Validation Agent** — Deterministic database checks against SQLite inventory (no LLM)
3. **Fraud Detection** — 14 weighted signals scored to a 0–100 composite risk score
4. **Approval Agent** — 6-step rule-based routing: critical reject → high risk reject → validation escalation → warning escalation → auto-approve → HITL
5. **Payment Agent** — Mock transaction handling + invoice history recording
6. **Explanation Agent** — Grok generates VP-readable decision summaries

Routing includes retry loops (up to 3 attempts with structured feedback) for extraction failures and `interrupt()`-based human-in-the-loop checkpoints for mid-range risk cases.

---

## Key Design Decision

LangGraph's `StateGraph` with conditional edges maps perfectly to this pipeline's cyclic structure — the self-correction loop (validate → retry → re-extract) and multi-path approval routing are first-class concepts, not bolted-on workarounds. The `interrupt()` mechanism genuinely pauses the state machine and persists the checkpoint via `MemorySaver`, enabling real HITL review in the Streamlit UI rather than simulating it with random decisions. Grok is accessed via `langchain-xai` (`ChatXAI` with `with_structured_output()`), making model swaps a one-line config change. Business rules (stock checks, math verification, duplicate detection) are pure Python with SQLite — the LLM is only called where it adds real value: unstructured text parsing, risk narratives, and plain-English summaries.

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/sami2919/Invoice-Processing-Automation.git && cd Invoice-Processing-Automation
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and set XAI_API_KEY=xai-your-key-here

# 3. Initialise the database (auto-seeds inventory + vendors)
python -c "from src.database import init_db; init_db()"
```

---

## Usage

**Process a single invoice (CLI)**
```bash
python main.py --invoice_path data/invoices/invoice_1001.txt
```

**Batch processing with CSV export**
```bash
python main.py --batch data/invoices/ --auto-approve
# Outputs: batch_results_YYYYMMDD_HHMMSS.csv
```

**Auto-approve mode (no HITL prompts — useful for CI)**
```bash
python main.py --invoice_path data/invoices/invoice_1001.txt --auto-approve
```

**Streamlit UI (recommended for demos)**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**Docker**
```bash
docker build -t invoice-ai .
docker run -p 8501:8501 -e XAI_API_KEY=xai-your-key invoice-ai
# Open http://localhost:8501
```

---

## Features

| Feature | Detail |
|---------|--------|
| **Multi-format ingestion** | PDF (pdfplumber), CSV, JSON, XML, TXT |
| **Self-correction extraction** | Grok extraction failures feed back into a retry loop (up to 3 attempts) with targeted error context |
| **OCR artifact handling** | Letter-O vs zero, mangled decimals — corrected at extraction time |
| **Fuzzy item matching** | `difflib.SequenceMatcher` at 0.8 threshold maps "Widget A" → "WidgetA" |
| **Composite fraud scoring** | 14 weighted signals, score 0–100, three tiers: auto-approve / flag / block |
| **Human-in-the-loop** | LangGraph `interrupt()` + `MemorySaver` checkpointing — genuine pause/resume via Streamlit review panel |
| **Duplicate detection** | Invoice number cross-checked against `invoice_history` table (recorded on both approval and rejection) |
| **Aggregate stock check** | Sums quantities per item across all line items before comparing to inventory stock |
| **Currency enforcement** | Flags non-USD invoices for review |
| **Price variance detection** | >10% deviation from catalog price flagged as validation warning |
| **VP-readable explanations** | Grok generates plain-English summaries of every decision |
| **Batch processing** | Directory scan with stem-based dedup, progress bar, colour-coded results, CSV export |
| **Full audit trail** | Every agent action recorded with timestamp, duration, and confidence |
| **Streamlit dashboard** | 4-tab UI: process, batch, analytics, audit trail + sidebar with settings and pipeline diagram |

---

## Business Impact

| Metric | Manual | Automated |
|--------|--------|-----------|
| Cost per invoice | ~$15 | ~$2.50 |
| Cycle time | 5 days | < 30 seconds |
| Error rate | ~30% | < 5% (AI validation + human oversight) |
| Duplicate detection | Reactive (after payment) | Proactive (before processing) |

**At 1,000 invoices/month:** ~$150,000 annual savings vs manual processing.

Fraud detection prevents duplicate payments (~2% overpayment rate). At $5M annual AP spend, that's $100,000+ in prevented overpayments. Combined with validation accuracy improvements, the system pays for itself within weeks of deployment.

---

## Test Matrix

| Invoice | Tests | Expected Result |
|---------|-------|-----------------|
| `INV-1001.txt` | Clean happy path | Approved |
| `INV-1002.txt` | Stock mismatch (GadgetX qty 20, stock 5) | Rejected |
| `INV-1003.txt` | Fraud signals: urgency language, wire transfer, "Fraudster LLC", zero-stock item | Rejected, high risk |
| `INV-1004.json` + `INV-1004_revised.json` | Duplicate invoice detection | Second flagged as duplicate |
| `INV-1005.json` | Unknown vendor + stock mismatch (GadgetX qty 8, stock 5) | Rejected |
| `INV-1006.csv` | Clean CSV format ingestion | Approved |
| `INV-1007.csv` | Stock mismatch (WidgetA qty 20, stock 15) | Rejected |
| `INV-1008.txt` | Unknown items (SuperGizmo, MegaSprocket) — all items unknown | Rejected |
| `INV-1009.json` | Negative quantity, empty vendor, null due date | Rejected |
| `INV-1010.txt` | Price variance (WidgetA at $250 regular + $300 rush) | Flagged for review |
| `INV-1011.pdf` | PDF ingestion via pdfplumber | Approved |
| `INV-1012.pdf` | OCR artifacts ("2O26", "3,500.O0", "Widget A") | Flagged — extraction resilience |
| `INV-1013.json` | Aggregate qty check (WidgetA 15+5+2=22, stock 15) | Rejected — stock mismatch |
| `INV-1014.xml` | Non-USD currency (EUR) | Flagged — currency warning |
| `INV-1015.csv` | Multi-line CSV with tax column | Approved |
| `INV-1016.json` | Unknown item (WidgetC) alongside valid items | Flagged for review |

Run with:
```bash
python main.py --batch data/invoices/ --auto-approve
```

---

## Test Suite

```bash
pytest tests/ -v
# 180 tests across 12 modules — validation, fraud scoring, approval routing,
# HITL auto-decide, batch dedup, rejection recording, extraction, pipeline routing
```

Tests are fully isolated — an in-memory SQLite database is spun up per test session and all Grok API calls are mocked. No API key required to run the test suite.

---

## What I'd Build Next

Framed through Galatiq's forward-deployment model:

- **SAP / NetSuite connector** — map invoice fields to purchase order IDs for 3-way matching (PO → receipt → invoice)
- **Grok Vision for scanned invoices** — bypass OCR entirely for paper invoices, passing the raw image to Grok's multimodal endpoint
- **A2A protocol** — expose the pipeline as an Agent-to-Agent endpoint so procurement systems can submit invoices programmatically
- **Slack / Teams notifications** — push HITL review requests directly to the approver's channel with one-click approve/reject
- **Fine-tuned extraction model** — train a lighter model on the client's historical invoice corpus to eliminate Grok calls for routine formats (80%+ of volume)
- **PostgreSQL** — replace SQLite with Postgres for multi-user concurrency and persistent audit trail
- **LangGraph Cloud** — deploy to LangGraph's hosted runtime for scalable HITL with email/Slack interrupts and persistent thread storage

---

## Personal Context

Having worked on AI systems for real-world operations, I expanded the case assumptions based on patterns observed in actual AP workflows — duplicate invoices submitted under slightly different formats, multi-currency procurement, aggregate quantity mismatches across split purchase orders, and the critical need for explainable risk scoring that finance teams actually trust. The test invoice set was designed to surface these real-world failure modes rather than just the obvious happy path. The 6-step approval routing and warning-aware HITL auto-decide logic reflect the nuance required when deploying AI into regulated financial processes where false rejections are as costly as false approvals.

---

## Project Structure

```
Invoice-Processing-Automation/
├── app.py                    # Streamlit UI (4 tabs: Process, Batch, Analytics, Audit)
├── main.py                   # CLI entry point (single + batch modes)
├── src/
│   ├── agents/
│   │   ├── extraction.py     # Grok JSON extraction + OCR correction + self-correction loop
│   │   ├── validation.py     # Deterministic DB checks (no LLM)
│   │   ├── fraud.py          # 14-signal composite risk scorer + Grok narrative
│   │   ├── approval.py       # 6-step routing + LangGraph interrupt() for HITL
│   │   ├── payment.py        # Mock payment + invoice history recording
│   │   └── explanation.py    # Grok VP-level decision summary
│   ├── models/
│   │   ├── invoice.py        # ExtractedInvoice, LineItem, FraudResult, ApprovalDecision
│   │   ├── state.py          # InvoiceState TypedDict (LangGraph state)
│   │   └── audit.py          # ProcessingRecord, BatchResult
│   ├── llm/
│   │   └── grok_client.py    # ChatXAI wrapper: assess() + get_structured_llm()
│   ├── tools/
│   │   ├── inventory_db.py   # SQLite queries (vendor, item, stock, duplicate checks)
│   │   ├── file_parser.py    # Multi-format invoice parser
│   │   ├── payment_api.py    # Mock payment API
│   │   └── pdf_extractor.py  # pdfplumber wrapper
│   ├── pipeline.py           # LangGraph StateGraph assembly + process_invoice()
│   ├── config.py             # pydantic-settings (all thresholds + API config)
│   ├── database.py           # Schema creation + seed data (WAL mode)
│   └── theme.py              # Dark executive dashboard styling
├── tests/
│   ├── conftest.py           # Isolated DB + invoice fixtures
│   ├── test_extraction.py    # Extraction: parse formats, fuzzy match, OCR
│   ├── test_validation.py    # Validation: each check function
│   ├── test_fraud.py         # Fraud: each weighted signal
│   ├── test_approval.py      # Approval: 6-step routing logic
│   ├── test_auto_decide.py   # HITL auto-decide: risk + warning handling
│   ├── test_batch_dedup.py   # Batch: stem-based file deduplication
│   ├── test_rejection_recording.py  # Rejection: invoice history recording
│   ├── test_grok_client.py   # LLM client: ChatXAI wrapper
│   ├── test_pdf_extractor.py # PDF: pdfplumber extraction
│   ├── test_theme.py         # Theme: styling helpers
│   ├── test_pipeline.py      # Pipeline: routing, compilation, recursion limit
│   └── test_integration.py   # E2E: happy path, rejection, HITL, retry, batch
├── data/invoices/            # 21 test invoices (TXT, JSON, CSV, XML, PDF)
├── Dockerfile                # Python 3.11-slim, Streamlit on 8501
├── requirements.txt
├── pyproject.toml
└── .env.example              # API key + threshold configuration
```
