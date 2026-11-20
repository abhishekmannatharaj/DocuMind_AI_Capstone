# DocuMind AI — Review 1 Presentation Script & Defense Playbook
## SAP Code Unnati 5.0 Capstone Review
**Total Marks: 100** | **Target Score: 100/100**  
**Presenters:** Abhishek Mannatharaj & Devika G  
**Institution:** Gopalan College of Engineering and Management, Bengaluru  

---

## Rubric Criterion Mapping & Time Allocation (Total: 10–12 Minutes)

| Rubric Criterion | Marks | Speaker | Allocated Time | Core Evaluator Focus |
|---|---|---|---|---|
| **1. Problem Statement & Relevance** | 15 | **Abhishek** | 2.0 min | Clear business problem, target users, SAP MM/FI process relevance, scoped objectives. |
| **2. Solution Design & SAP Architecture** | 20 | **Devika** | 2.5 min | Architecture, SAP BTP/HANA/S4HANA mapping, 3-way matching data model, process flow. |
| **3. Innovation & Impact on Target Users** | 15 | **Abhishek** | 1.5 min | Novelty, agentic AI vs static OCR, measurable metrics, STP acceleration. |
| **4. Implementation Progress & Demo** | 20 | **Abhishek & Devika** | 3.5 min | Working system live demo, upload, confidence scores, SAP JSON export, RAG citations. |
| **5. Domain Knowledge** | 15 | **Devika** | 1.5 min | SAP concepts (`MIRO`, `EKKO`, `RBKP`, `BAPI`, GST compliance). |
| **6. Presentation & Q&A Defense** | 15 | **Both** | 2.0 min | Team cohesion, distinct individual contributions, crisp defense. |

---

## Detailed Presentation Script

### Slide 1: Title & Introduction (0:00 – 0:30) — *Speaker: Abhishek*
> **Abhishek:**  
> "Good morning, respected evaluators and panel members. We are Abhishek Mannatharaj and Devika G from Gopalan College of Engineering and Management. Today, under SAP Code Unnati 5.0, we present our capstone project: **DocuMind AI: Intelligent Document Understanding for Invoices, Purchase Orders, and Forms**."

---

### Slide 2: Criterion 1 — Problem Statement & Enterprise SAP Relevance (15 Marks) (0:30 – 2:00) — *Speaker: Abhishek*
> **Abhishek:**  
> "In enterprise resource planning, Accounts Payable is the financial backbone of procurement. In a typical SAP S/4HANA deployment, incoming vendor invoices arrive as unstructured PDFs, scanned images, and emails in various languages.  
> 
> Currently, enterprise AP teams face three critical pain points:  
> 1. **Manual Entry Overhead:** Entering invoice data into SAP transaction `MIRO` takes 10 to 15 minutes per document, resulting in backlogs and human transposition errors.  
> 2. **3-Way Matching Bottlenecks:** Reconciling invoice line items against SAP Purchase Orders (`EKPO`) and Goods Receipts manually causes delayed supplier payments and missed cash discounts.  
> 3. **Tax & Multilingual Friction:** International vendors submit invoices in Spanish or German, and Indian vendors require strict 15-digit GSTIN compliance.  
> 
> **Our Objective:** Build an autonomous, agentic system that ingests multi-format invoices, extracts structured data with confidence scoring, performs automated 3-way reconciliation against SAP ERP records, and produces SAP Posting-Ready payloads with an interactive Human-in-the-Loop review screen."

---

### Slide 3: Criterion 2 — Solution Design & SAP Architecture (20 Marks) (2:00 – 4:30) — *Speaker: Devika*
> **Devika:**  
> "Thank you, Abhishek. Let me walk you through the Solution Architecture and SAP Technology alignment.  
> 
> DocuMind AI is engineered around an **Enterprise Multi-Agent Pipeline orchestrated via LangGraph**:  
> 1. **Invoice Monitor Agent:** Simulates enterprise email inbox ingestion and file watching.  
> 2. **Extractor Agent:** Utilizes hybrid offline extraction via `pdfplumber` for digital PDFs and OCR fallback for scanned images, attaching per-field confidence scores.  
> 3. **Translation Agent:** Normalizes foreign language vendor documents into English while strictly preserving monetary and SKU values.  
> 4. **Validation Agent:** Validates data types, line-item math, and Indian statutory rules including GSTIN checksum matching against `rules.yaml`.  
> 5. **Business Validation Agent (SAP 3-Way Match):** Directly connects to our **SAP ERP Mock Service** running on FastAPI port 8001. It cross-references the invoice against Purchase Orders (`PO Records.json` representing SAP table `EKKO`/`EKPO`), Vendor Master (`vendors.json` representing SAP table `LFA1`), and SKU Master (`sku_master.json` representing `MARA`).  
> 6. **Reporting & Vector Subsystem:** Generates comprehensive HTML audit reports and indexes all invoice contexts into a **FAISS vector store** using 384-dimensional dense semantic embeddings.  
> 
> In a production SAP landscape, this platform sits on **SAP BTP (Business Technology Platform)**, utilizing the **SAP BTP Integration Suite** to post verified documents directly into S/4HANA via `BAPI_INCOMINGINVOICE_CREATE` or IDoc `INVOIC02`."

---

### Slide 4: Criterion 3 — Innovation & Measurable Business Impact (15 Marks) (4:30 – 6:00) — *Speaker: Abhishek*
> **Abhishek:**  
> "What makes DocuMind AI innovative compared to traditional OCR tools like ABBYY or standard cloud vision APIs?  
> 
> 1. **Zero Silent Failures with Granular Confidence Auditing:** Instead of opaque outputs, every extracted field carries an individual confidence metric. High-confidence invoices straight-through-process (STP) into SAP, while low-confidence fields are highlighted with visual badges for Human Review.  
> 2. **Free, Offline-Friendly Local Architecture:** Unlike systems tied to expensive proprietary APIs like Azure OpenAI, our implementation runs 100% locally with `sentence-transformers`, FAISS, and local LLMs (Ollama) or free API tiers (Groq/Gemini), guaranteeing zero deployment cost for reviewing and offline evaluation.  
> 3. **Measurable ROI:** Turnaround time drops from **12 minutes to under 2.5 seconds** per invoice, enabling over **70% Straight-Through Processing (STP)**."

---

### Slide 5: Criterion 4 — Implementation Progress & Live System Demo (20 Marks) (6:00 – 9:30) — *Live Demo by Both*
> **Abhishek:**  
> "Now let us demonstrate our working prototype, live. As you can see, both our **SAP ERP Mock Service on port 8001** and our **Streamlit Dashboard on port 8501** are actively running."

*(Abhishek shows Dashboard & Pipeline tab)*:
> **Abhishek:**  
> "On the Dashboard, we have a live overview of processed invoices showing Auto-Approved, Manual Review, and Discrepancy counts. Let me run the full pipeline across queued documents."  
*(Clicks 'Run Full Pipeline' -> shows multi-agent execution step-by-step)*:  
> "Notice the multi-agent state progression: Extractor -> Translator -> Validation -> SAP ERP Business Cross-Reference -> HTML Reporting."

*(Devika takes over to demonstrate the new Human-in-the-Loop & Custom Upload page)*:
> **Devika:**  
> "Now, let's explore our custom document upload and Human-in-the-Loop review feature.  
> I will upload an invoice directly. The system extracts all fields in real-time and displays **per-field AI confidence badges**: Green for high confidence (>90%), Amber for human review required.  
> Notice our Indian compliance feature: the vendor's 15-digit GSTIN is extracted and verified against statutory rules.  
> As a human auditor, I can modify any field, select 'Approve for SAP Posting', and click **Export SAP Posting-Ready JSON**.  
> This downloads an enterprise-ready JSON payload structured identically to the input parameters required by SAP's `BAPI_INCOMINGINVOICE_CREATE`."

*(Devika shows RAG Support Agent tab)*:
> **Devika:**  
> "Finally, let's query our DocuMind Support Agent. I'll ask: *'Why was invoice INV_EN_006 flagged for manual review?'*  
> The agent retrieves facts from the FAISS vector index and provides an answer accompanied by **page citation tags e.g. `[Page 1]`**, providing full audit traceability."

---

### Slide 6: Criterion 5 — Domain Knowledge & Review 2 Plan (15 Marks) (9:30 – 10:30) — *Speaker: Devika*
> **Devika:**  
> "To summarize our domain alignment:  
> We have mapped document understanding directly to SAP MM-FI integration:  
> - Purchase Order: `EKKO` / `EKPO`  
> - Vendor Master: `LFA1` with statutory tax code `STCD3` (GSTIN)  
> - Invoice Verification: `RBKP` / `RSEG`  
> 
> **Plan for Review 2:**  
> 1. Integrate SAP Cloud SDK to dispatch OData calls to a live SAP BTP S/4HANA sandbox.  
> 2. Implement an automated PO line-item matching algorithm using Levenshtein distance for fuzzy description matching.  
> 3. Provide an audit export report formatted as a standard SAP Spool / PDF document."

---

### Slide 7: Criterion 6 — Q&A Defense & Team Contributions (15 Marks) (10:30+) — *Both*
> **Abhishek:**  
> "In terms of individual contributions:  
> - **Abhishek:** Responsible for the agentic pipeline orchestration (LangGraph state machines), offline LLM/OCR fallback providers, FastMCP tool integration, and Streamlit UI design.  
> - **Devika:** Responsible for SAP ERP data modeling, Indian GSTIN compliance rules, 3-way matching business validation logic, FAISS vector indexer, and RAG retrieval pipeline with page citations.  
> 
> We are now ready for questions from the panel. Thank you!"
