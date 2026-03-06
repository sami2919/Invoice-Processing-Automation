# Invoice Processing AI

Multi-agent system built with **LangGraph + Grok** that automates invoice processing: ingestion → validation → fraud detection → approval → payment → explainability.

## Architecture

```
extract → validate → [retry loop | fraud_check] → approve → [payment | reject] → explain → END
```

- **Extraction**: Grok structured output with self-correction loop (up to 3 retries)
- **Validation**: 10 deterministic checks against SQLite (stock, math, duplicates, vendor approval)
- **Fraud detection**: 14 weighted signals aggregated to 0–100 risk score
- **Approval**: Auto-approve (low risk + small amount), HITL interrupt (medium), auto-reject (high risk)
- **Payment**: Mock payment API + invoice recording
- **Explanation**: VP-readable Grok summary

Key design choices:
- `interrupt()` for real HITL — genuinely pauses the graph, resumes on human decision
- Deterministic validation & fraud scoring — LLM only used for text extraction and narratives
- Self-correction loop with structured feedback when extraction fails validation

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# set XAI_API_KEY in .env
```

## Usage

```bash
# single invoice
python main.py --invoice_path=data/invoices/invoice_1001.txt

# batch mode
python main.py --batch=data/invoices/ --auto-approve --fresh

# streamlit dashboard
streamlit run app.py
```

## Tests

```bash
pytest -v
```

48 tests covering extraction, validation, fraud scoring, pipeline routing, and 5 end-to-end integration tests. All LLM calls are mocked — no API key needed to run tests.

## What I'd build next

- SAP/NetSuite connector for 3-way PO matching
- Grok Vision for scanned PDFs instead of pdfplumber
- PostgreSQL for production concurrency
- Slack notifications for HITL interrupts
- LangGraph Cloud for hosted runtime with persistent threads

## Project Structure

```
src/
  agents/          # extraction, validation, fraud, approval, payment, explanation
  models/          # Pydantic models (invoice, state, audit)
  tools/           # inventory_db, file_parser, payment_api, pdf_extractor
  llm/             # grok_client (ChatXAI wrapper)
  pipeline.py      # LangGraph StateGraph assembly
  database.py      # SQLite init + seed data
  config.py        # pydantic-settings
tests/             # pytest suite (conftest + 5 test files)
data/invoices/     # 21 test invoices (TXT, JSON, CSV, XML, PDF)
main.py            # CLI entry point
app.py             # Streamlit dashboard
```
