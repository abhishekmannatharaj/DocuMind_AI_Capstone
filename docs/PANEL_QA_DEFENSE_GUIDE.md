# DocuMind AI — Panel Q&A Defense Guide & Domain Cheat Sheet
## SAP Code Unnati 5.0 Capstone Review

This guide arms **Abhishek Mannatharaj** and **Devika G** with deep technical and SAP domain answers to score **100% in Criterion 5 (Domain Knowledge - 15 marks)** and **Criterion 6 (Presentation & Q&A - 15 marks)**.

---

## 1. Top SAP Enterprise Domain Questions & Answers

### Q1: "How does DocuMind AI connect to SAP? Which standard SAP transaction or BAPI does it replace?"
> **Answer:**  
> "DocuMind AI does not replace SAP; it automates the manual upstream process before SAP. In standard SAP ERP/S4HANA, an Accounts Payable accountant opens transaction **`MIRO`** (Enter Incoming Invoice) or **`FB60`** (Enter Incoming Invoice for FI). They manually type in the PO Number, Vendor ID, Invoice Amount, and check each line item.  
> DocuMind AI automates this by extracting data from the invoice, cross-referencing our SAP ERP Mock Service (simulating `EKKO`/`EKPO` for POs and `LFA1` for Vendors), and generating a **SAP Posting-Ready JSON payload**. In production on SAP BTP, this payload triggers **`BAPI_INCOMINGINVOICE_CREATE`** or an **IDoc `INVOIC02`** via SAP BTP Integration Suite."

### Q2: "What is SAP 3-Way Matching, and how does your code implement it?"
> **Answer:**  
> "SAP 3-Way Matching verifies three documents:  
> 1. **Purchase Order (PO):** What was ordered (`EKKO`/`EKPO`).  
> 2. **Goods Receipt (GR):** What was delivered (`MSEG`/`MKPF`).  
> 3. **Vendor Invoice:** What was billed.  
> 
> In our code (`backned/mcp/tools/business_validation_tool.py`), we implement this by comparing:  
> - **Quantity Tolerance:** Invoice quantity must not exceed PO open quantity (0% upward tolerance).  
> - **Price Tolerance:** Unit price variance is checked against a configurable threshold (±5% in `rules.yaml`).  
> - **Vendor Reconciliation:** Invoice vendor must match the PO vendor registered in the SAP Vendor Master (`LFA1`)."

### Q3: "How does your system handle Indian Statutory Compliance and GST?"
> **Answer:**  
> "In Indian taxation, invoices must comply with the Goods and Services Tax (GST) framework. In `rules.yaml`, we defined statutory rules:  
> 1. **15-digit GSTIN Regex Verification:** `^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$` where the first 2 digits represent State Code, next 10 digits are Vendor PAN, followed by Entity number, 'Z' by default, and a checksum character.  
> 2. **Mock ERP Cross-Check:** Our vendor master fixture (`vendors.json`) stores the vendor's official GSTIN (simulating SAP table field `LFA1-STCD3`). If the invoice GSTIN doesn't match the SAP master record, the document is flagged for Human Review."

---

## 2. Technical Architecture & AI Questions

### Q4: "Why use LangGraph instead of standard LangChain chains or sequential Python scripts?"
> **Answer:**  
> "Standard chains are strictly acyclic and linear (DAGs), which cannot handle complex real-world workflows that require cyclic loops, conditional branching, and state rollbacks.  
> LangGraph models our pipeline as a **State Machine**. For instance, in our RAG Reflection stage (`backned/agents/rag_agents/reflection_agent.py`), the Reflection Agent evaluates the answer quality; if quality is below threshold, the graph loops back and triggers regeneration. LangGraph also maintains typed Pydantic state persistence across nodes."

### Q5: "How did you eliminate the paid Azure OpenAI dependency?"
> **Answer:**  
> "Originally, many enterprise prototypes rely on Azure OpenAI GPT-4o Vision and OpenAI embeddings, which require paid API credits and fail when internet is restricted.  
> We architected **`backned/services/llm_provider.py`**:  
> 1. Digital PDFs are parsed with **`pdfplumber`** locally.  
> 2. Scanned images utilize local OCR with **`pytesseract` / Pillow**.  
> 3. Embeddings are generated using **`sentence-transformers` (`all-MiniLM-L6-v2`)** producing 384-dimensional dense vectors indexed into FAISS.  
> 4. We support free/local LLM backends via **Ollama (`llama3.2`)**, **Groq (free LLaMA 3.3)**, or **Gemini API**, backed by a deterministic rule-based extractor so the demo never crashes even completely offline."

### Q6: "How do page citations work in your RAG subsystem?"
> **Answer:**  
> "When documents and audit reports are indexed into our FAISS vector store by `vector_indexer_tool.py`, each text chunk retains metadata including document filename and page boundary tags e.g. `[Page 1]`. When the Semantic Retriever retrieves top chunks and the Response Synthesizer crafts the final answer, it is prompted to explicitly append `[Page 1]` citations next to factual claims, giving enterprise auditors verifiable proof."

---

## 3. Evaluator Trap Questions & Quick Counters

| Trap Question | Bad Answer | Winning Answer (Say this!) |
|---|---|---|
| *"Is this just a simple Streamlit wrapper on an LLM?"* | *"No, we used LangChain and Streamlit."* | *"Not at all. The Streamlit UI is only the presentation layer. The core is an autonomous 6-agent LangGraph state machine with an enterprise Mock ERP service, FastMCP tool abstraction, FAISS vector database, and custom Human-in-the-Loop review mechanisms."* |
| *"What happens if an invoice has a completely new layout you haven't seen?"* | *"It might fail."* | *"Our hybrid extraction strategy first attempts coordinate and table-based extraction via pdfplumber. If any mandatory field falls below our 85% confidence threshold, our Human-in-the-Loop screen automatically flags it in amber for the AP clerk to verify before SAP ingestion."* |
| *"Who did what in this project?"* | *"We worked together on everything."* | *"We had clear ownership: Abhishek focused on the LangGraph agent orchestration, FastMCP server, offline LLM/OCR providers, and Streamlit UI. Devika designed the SAP ERP data models, 3-way matching business validation logic, GSTIN compliance, and FAISS RAG indexing with page citations."* |
