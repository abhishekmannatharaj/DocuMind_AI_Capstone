## Demo 

https://github.com/user-attachments/assets/3a9aa479-ab76-47d8-83c3-cb8762617b26
 
 # DocuMind AI — Intelligent Document Understanding for Invoices, Purchase Orders & Forms
### SAP Code Unnati 5.0 Capstone Project
**Authors:** Abhishek Mannatharaj & Devika G | Dept. of CSE, Gopalan College of Engineering and Management, Bengaluru

An intelligent document understanding and invoice auditing platform that automates the entire document-processing lifecycle, from extraction and translation to validation, compliance checks, and ERP cross-referencing. It processes documents from PDFs, DOCX files, and scanned images, extracts structured invoice data, validates fields against SAP ERP records (PO & Vendor Master) including Indian GSTIN and INR currency compliance, generates audit reports, and enables users to analyze documents through an AI interface. Built with **LangGraph**, **FastMCP**, **FAISS**, and **Streamlit**, featuring offline/local models and free LLM providers (Ollama, Groq, Gemini).

> ### 🏆 SAP Code Unnati 5.0 Review Documentation (Rubric Aligned)
> - 📘 **[Enterprise SAP Architecture Specification](docs/SAP_ARCHITECTURE.md)**: Deep dive into SAP S/4HANA (MM/FI), SAP BTP, 3-Way Matching (`EKKO`, `EKPO`, `MSEG`, `RBKP`, `RSEG`), and SAP Data Models.
> - 🎤 **[Review 1 Presentation Script (100-Mark Rubric)](docs/REVIEW_1_PRESENTATION_SCRIPT.md)**: Turn-by-turn speaking script dividing presentation between **Abhishek Mannatharaj** & **Devika G**.
> - 🛡️ **[Panel Q&A Defense Guide & Domain Knowledge Sheet](docs/PANEL_QA_DEFENSE_GUIDE.md)**: Anticipated evaluator questions on `MIRO`, BAPIs, tolerances, OCR, and LangGraph vs traditional chains.

---

## Table of Contents

1. [Overview](#overview)
2. [Solution Architecture](#solution-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Pipeline Agents](#pipeline-agents)
6. [RAG Subsystem](#rag-subsystem)
7. [MCP Server & Tools](#mcp-server--tools)
8. [SAP ERP Mock Service](#sap-erp-mock-service)
9. [Validation Rules & Indian GSTIN](#validation-rules--indian-gstin)
10. [Observability](#observability)
11. [Frontend & Human Review](#frontend--human-review)
12. [Invoice Dataset](#invoice-dataset)
13. [Getting Started](#getting-started)
14. [Configuration](#configuration)

---

## Overview

DocuMind AI provides automated, intelligent understanding of financial documents:

- Detects new invoice files or allows direct manual document upload.
- Extracts structured data from PDF, DOCX, and scanned image files using `pdfplumber` and offline OCR / LLM fallback.
- Validates completeness and data quality against configurable business rules, supporting INR and Indian GSTIN formatting.
- Cross-references document data against the **SAP ERP Mock Service (PO & Vendor Master)**.
- Provides interactive Human-in-the-Loop review for low-confidence fields with approval, editing, and SAP Posting-Ready JSON export.
- Indexes document chunks into a FAISS vector store with page-level citations (`[Page X]`) for conversational Q&A.

---

## Solution Architecture

```
+-------------------------------------------------------------------------+
|                    Streamlit Frontend (fronted/app.py)                  |
|  Dashboard . Pipeline Runner . Report Viewer . RAG Chat Interface       |
+----------------------------------+--------------------------------------+
                                   | orchestrated by
                                   v
+-------------------------------------------------------------------------+
|                  Pipeline Runner (fronted/pipeline_runner.py)           |
|                                                                         |
|  Stage 1          Stage 2          Stage 3          Stage 4             |
|  Invoice   -->   Extractor  -->  Translation  -->  Validation           |
|  Monitor         Agent            Agent             Agent               |
|  Agent                                                                  |
|                                   Stage 5          Stage 6              |
|                              Business Validation > Reporting            |
|                              Agent                  Agent               |
+--------------------------------+-----------------------------------------+
                                 | calls via MCP
                                 v
+-------------------------------------------------------------------------+
|               FastMCP Server (backned/mcp/server/mcp_server.py)         |
|                                                                         |
|  invoice_watcher . data_harvester . lang_bridge . data_completeness     |
|  business_validation . insight_reporter                                 |
|  vector_indexer . semantic_retriever . chunk_ranker . response_synth    |
+------------------------------+------------------------------------------+
                               | backed by
          +--------------------+--------------------+
          v                    v                    v
   Azure OpenAI           FAISS Vector         Mock ERP API
   (LLM + Embeddings)     Store                (FastAPI :8001)
```

The system follows a **multi-agent, tool-augmented** architecture. Each pipeline stage is an independent **LangGraph** state machine that communicates by passing structured Pydantic state. Agents do not call LLMs or deal with files directly — they invoke **MCP tools** via a central FastMCP server, which enforces a clean tool abstraction layer.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM / Embeddings | Azure OpenAI (GPT-4o, text-embedding-3-small) |
| Tool protocol | FastMCP (Model Context Protocol) |
| Vector store | FAISS (faiss-cpu) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Document parsing | pdfplumber, python-docx, Pillow |
| Observability | LangFuse |
| Configuration | Pydantic, python-dotenv, PyYAML |

---

## Project Structure

```
Invoice_Auditor/
+-- backned/                        # All backend logic
|   +-- agents/                     # LangGraph pipeline agents
|   |   +-- invoice_monitor_agent.py
|   |   +-- extractor_agent.py
|   |   +-- translation_agent.py
|   |   +-- validation_agent.py
|   |   +-- business_validation_agent.py
|   |   +-- reporting_agent.py
|   |   +-- rag_agents/             # 5 RAG sub-agents
|   |       +-- indexing_agent.py
|   |       +-- retrieval_agent.py
|   |       +-- augmentation_agent.py
|   |       +-- generation_agent.py
|   |       +-- reflection_agent.py
|   +-- api/
|   |   +-- erp_mock_api.py         # FastAPI mock ERP server
|   +-- config/
|   |   +-- settings.py             # Centralised env-var config
|   |   +-- rules.yaml              # Validation & business rules
|   |   +-- logging_setup.py
|   |   +-- observability.py        # LangFuse integration
|   +-- mcp/
|   |   +-- server/mcp_server.py    # FastMCP server
|   |   +-- tools/                  # Pipeline tool implementations
|   |   |   +-- invoice_watcher_tool.py
|   |   |   +-- data_harvester_tool.py
|   |   |   +-- lang_bridge_tool.py
|   |   |   +-- data_completeness_checker_tool.py
|   |   |   +-- business_validation_tool.py
|   |   |   +-- insight_reporter_tool.py
|   |   |   +-- rag_tools/
|   |   |       +-- vector_indexer_tool.py
|   |   |       +-- semantic_retriever_tool.py
|   |   |       +-- chunk_ranker_tool.py
|   |   |       +-- response_synthesizer_tool.py
|   |   +-- clients/mcp_clients.py  # Sync MCP tool invocation helper
|   +-- models/                     # Pydantic data models
|   +-- services/rag_pipeline.py    # RAG orchestration service
|   +-- data/                       # Invoice files & ERP mock data
|       +-- incoming/               # Invoice attachments (.pdf, .docx, .png)
|       +-- ERP_mockdata/           # JSON fixtures for mock ERP
+-- fronted/                        # Streamlit frontend
|   +-- app.py                      # Main Streamlit application
|   +-- pipeline_runner.py          # Orchestrates all 6 pipeline agents
+-- outputs/
|   +-- reports/                    # Generated HTML audit reports
+-- logs/                           # Application log files
+-- requirements.txt
+-- .env                            # API keys and environment settings
```

---

## Pipeline Agents

The full pipeline is composed of **six sequential LangGraph agents**. Each agent is a self-contained state machine with typed state, LangGraph nodes, and conditional edges.

### Stage 1 — Invoice Monitor Agent

**File:** `backned/agents/invoice_monitor_agent.py`

Scans the `data/incoming/` directory for new `.meta.json` email files that have not yet been processed. Invokes the `invoice_watcher` MCP tool, records new email IDs, and persists the processed list to `processed_emails.json` to avoid re-processing on subsequent runs.

**LangGraph flow:** `scan_inbox -> evaluate_results -> mark_processed -> END`

---

### Stage 2 — Extractor Agent

**File:** `backned/agents/extractor_agent.py`

Iterates over every new email and its attachment list. Invokes the `data_harvester` MCP tool for each file, which dispatches to the correct parser:

- **PDF** — `pdfplumber` for text and table extraction.
- **DOCX** — `python-docx` for paragraph and table extraction.
- **PNG / image** — Azure OpenAI Vision (GPT-4o) for OCR.

Returns structured `ExtractionResult` objects containing the raw text, extracted fields (invoice number, vendor, line items, totals), and detected language.

**LangGraph flow:** `extract_all -> validate_extractions -> END`

---

### Stage 3 — Translation Agent

**File:** `backned/agents/translation_agent.py`

Accepts all extraction results and translates any non-English invoice content into English via the `lang_bridge` MCP tool (backed by an Azure OpenAI LLM call). Preserves the original text alongside the translated version and attaches a **translation confidence score**.

English invoices pass through unchanged with a confidence of 1.0.

**LangGraph flow:** `translate_all -> verify_translations -> END`

---

### Stage 4 — Validation Agent

**File:** `backned/agents/validation_agent.py`

Validates each translated invoice against the rules defined in `rules.yaml` using the `data_completeness_checker` MCP tool. Checks:

- Presence of all **required header fields** (invoice_no, invoice_date, vendor_id, currency, total_amount).
- Presence of all **required line-item fields** (item_code, description, qty, unit_price, total).
- Correct **data types** for each field.
- **Currency** is in the accepted list (USD, EUR, INR, GBP).
- **Line-item math** consistency: qty x unit_price ≈ total.

Returns a `ValidationResult` with the list of missing fields, type errors, and a completeness score.

**LangGraph flow:** `validate_all -> summarize_validation -> END`

---

### Stage 5 — Business Validation Agent

**File:** `backned/agents/business_validation_agent.py`

Cross-references the extracted invoice data against the **Mock ERP system** using the `business_validation` MCP tool. Checks:

- **Purchase Order** lookup — does the referenced PO exist?
- **Vendor** lookup — does the vendor ID / name match ERP records?
- **SKU / item codes** — do all line items exist in the item master?
- **Price tolerances** — unit prices within +/-5% of ERP contract prices.
- **Quantity match** — quantities must match the PO exactly (0% tolerance).
- **Tax tolerances** — tax amounts within +/-2% of expected values.
- **Line-item math** — qty x unit_price ≈ total within +/-1%.

Returns a `BusinessValidationResult` with discrepancy details and an overall business validation score.

**LangGraph flow:** `business_validate_all -> summarize_business_validation -> END`

---

### Stage 6 — Reporting Agent

**File:** `backned/agents/reporting_agent.py`

Correlates all results from Stages 3-5 (translation, data validation, business validation) for each invoice and invokes the `insight_reporter` MCP tool to generate a styled **HTML audit report** saved to `outputs/reports/`.

Each report includes:

- Invoice header summary and translation details.
- Data completeness check results (field-by-field).
- ERP cross-reference results (PO, vendor, line-item comparisons).
- An overall **recommendation**: `Auto Approve`, `Manual Review`, or `Reject`.
  - **Auto-Approve** requires zero discrepancies and translation confidence >= 0.95.
  - **Reject** triggers if currency is invalid or critical fields are missing.
  - All other cases lead to **Manual Review**.

**LangGraph flow:** `generate_reports -> summarize_reports -> END`

---

## RAG Subsystem

A five-agent **Retrieval-Augmented Generation** pipeline allows users to ask natural-language questions about any processed invoice. It maintains in-session conversation memory (sliding window of 20 turns).

**Service entry point:** `backned/services/rag_pipeline.py`

| Agent | File | Responsibility |
|---|---|---|
| Indexing | `rag_agents/indexing_agent.py` | Parses all invoice documents, chunks text, embeds with Azure OpenAI, and upserts into the FAISS vector store (`rag_store/invoice_index.faiss`). |
| Retrieval | `rag_agents/retrieval_agent.py` | Embeds the user query and performs a top-K FAISS nearest-neighbour search. |
| Augmentation | `rag_agents/augmentation_agent.py` | Re-ranks and filters retrieved chunks for relevance using the `chunk_ranker` MCP tool. |
| Generation | `rag_agents/generation_agent.py` | Constructs a prompt from the ranked context and conversation history, then calls the LLM to produce a grounded answer. |
| Reflection | `rag_agents/reflection_agent.py` | Evaluates answer quality and relevance; flags low-confidence responses. |

**RAG flow:** `Indexing -> Retrieval -> Augmentation -> Generation -> Reflection`

---

## MCP Server & Tools

All side-effectful operations (file I/O, LLM calls, ERP API calls, vector store writes) are encapsulated as **MCP tools** registered on a `FastMCP` server instance.

**Server:** `backned/mcp/server/mcp_server.py`

Agents call tools via the synchronous helper `invoke_tool_sync()` in `backned/mcp/clients/mcp_clients.py`, which wraps the async MCP call in a synchronous event loop.

| Tool Name | Implementation | Purpose |
|---|---|---|
| `invoice_watcher` | `invoice_watcher_tool.py` | Scans `data/incoming/` for new unprocessed email files |
| `data_harvester` | `data_harvester_tool.py` | Parses PDF / DOCX / image attachments and extracts structured fields |
| `lang_bridge` | `lang_bridge_tool.py` | Translates invoice text into English via LLM |
| `data_completeness_checker` | `data_completeness_checker_tool.py` | Validates field presence, types, and currency rules |
| `business_validation` | `business_validation_tool.py` | Cross-checks against Mock ERP (PO, vendor, SKU, price tolerances) |
| `insight_reporter` | `insight_reporter_tool.py` | Generates HTML audit report with final recommendation |
| `vector_indexer` | `rag_tools/vector_indexer_tool.py` | Embeds and stores invoice chunks in FAISS |
| `semantic_retriever` | `rag_tools/semantic_retriever_tool.py` | Retrieves top-K relevant chunks for a query |
| `chunk_ranker` | `rag_tools/chunk_ranker_tool.py` | Re-ranks chunks by relevance score |
| `response_synthesizer` | `rag_tools/response_synthesizer_tool.py` | Synthesizes a grounded answer from ranked context |

---

## Mock ERP API

**File:** `backned/api/erp_mock_api.py`

A lightweight **FastAPI** application (default port **8001**) that simulates an enterprise ERP system. Data is served from JSON fixtures in `backned/data/ERP_mockdata/`.

| Endpoint | Description |
|---|---|
| `GET /api/erp/po/{po_number}` | Purchase Order with line items |
| `GET /api/erp/vendor/{vendor_id}` | Vendor master record |
| `GET /api/erp/sku/{item_code}` | SKU / item master record |
| `GET /api/erp/po` | List all purchase orders (debug) |
| `GET /api/erp/vendors` | List all vendors (debug) |
| `GET /api/erp/skus` | List all SKUs (debug) |

Start the ERP server before running the pipeline:

```bash
python -m backned.api.erp_mock_api
```

---

## Validation Rules

All validation logic is driven by `backned/config/rules.yaml`. Key sections:

| Section | Description |
|---|---|
| `required_fields` | Mandatory header and line-item fields |
| `data_types` | Expected Python type for each field |
| `tolerances` | Price (+/-5%), quantity (0%), tax (+/-2%), line-item math (+/-1%) deviation limits |
| `accepted_currencies` | USD, EUR, INR, GBP |
| `currency_symbol_map` | Symbol-to-ISO-code mapping ($, EUR, INR, GBP) |
| `validation_policies` | Actions for missing fields, total mismatches, invalid currency; auto-approve confidence threshold (0.95) |

---

## Observability

The system integrates **LangFuse** for full-pipeline tracing.

- Every `run_full_pipeline()` call produces a top-level LangFuse **trace** named `invoice-auditor-pipeline`.
- Each pipeline stage emits a child **span** linked to the trace.
- Every RAG `query()` produces a trace named `rag-pipeline-query` with child spans for each RAG stage.
- All Azure OpenAI LLM calls are automatically captured as **generation** events.

Configure via `.env`:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true
```

---

## Frontend

**File:** `fronted/app.py`

A **Streamlit** application with a dark-themed sidebar navigation providing four views:

| View | Description |
|---|---|
| **Dashboard** | Overview of all processed invoices with status indicators and quick-filter by recommendation |
| **Run Pipeline** | Trigger the full 6-stage pipeline with live stage-by-stage progress and log streaming |
| **Reports** | Browse and view generated HTML audit reports inside the app |
| **RAG Chat** | Conversational Q&A over all indexed invoices, powered by the RAG pipeline |

Start the frontend:

```bash
cd fronted
streamlit run app.py
```

---

## Invoice Dataset

The `backned/data/incoming/` directory contains synthetic, language-diverse invoice files simulating real email attachments:

| File | Language | Format | Notes |
|---|---|---|---|
| `INV_EN_001.pdf` | English | PDF | Standard invoice |
| `INV_EN_002.pdf` | English | PDF | Standard invoice |
| `INV_ES_003.pdf` | Spanish | PDF | Tests translation pipeline |
| `INV_DE_004.docx` | German | DOCX | Tests DOCX extraction and translation |
| `INV_EN_005_scan.png` | English | PNG | Tests OCR (handwritten-style scan) |
| `INV_EN_006_malformed.pdf` | English | PDF | Missing invoice number and currency — tests validation logic |

Each invoice has a `.meta.json` sidecar file that simulates an email header:

```json
{
  "sender": "accounts@globallogistics.com",
  "subject": "Invoice INV-1001 for PO-1001",
  "received_timestamp": "2025-03-14T09:32:00Z",
  "language": "en",
  "attachments": ["INV_EN_001.pdf"]
}
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- An Azure OpenAI resource with a GPT-4o deployment and a text-embedding model deployment.
- (Optional) A LangFuse account for observability.

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
# source .venv/bin/activate          # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
# Azure OpenAI
AZURE_AI_ENDPOINT=https://<your-resource>.openai.azure.com/
API_KEY=<your-api-key>
API_VERSION=2024-05-01-preview
DEPLOYMENT_NAME=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

# LangFuse (optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true
```

### Running the Application

```bash
# Terminal 1 — start the Mock ERP API
python -m backned.api.erp_mock_api

# Terminal 2 — start the Streamlit frontend
cd fronted
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

To run only the backend pipeline (headless):

```bash
python -m backned.agents.reporting_agent
```

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | API keys, Azure OpenAI settings, LangFuse credentials |
| `backned/config/settings.py` | Loads `.env` and exposes constants to all modules |
| `backned/config/rules.yaml` | All validation rules, tolerances, and policies |
| `backned/config/logging_setup.py` | Logging configuration (outputs to `logs/invoice_auditor.log`) |
| `backned/config/observability.py` | LangFuse client, `@observe` decorator, callback handler |
