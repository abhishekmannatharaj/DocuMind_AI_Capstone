"""FastMCP Server — Invoice Auditor MCP Server.

Exposes all Invoice Auditor pipeline tools (extraction, translation,
validation, business validation, reporting, and RAG) as MCP tools via
the FastMCP framework.

Tool *implementations* remain in ``backned/mcp/tools/`` — only imported here.

Usage
-----
Run directly::

    python -m backned.mcp.server.mcp_server

Or via uvicorn / stdio transport depending on FastMCP config.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

#  Pipeline tools (implementations in backned/mcp/tools/) 
from backned.mcp.tools.invoice_watcher_tool import invoice_watcher_tool
from backned.mcp.tools.data_harvester_tool import data_harvester_tool
from backned.mcp.tools.lang_bridge_tool import lang_bridge_tool
from backned.mcp.tools.data_completeness_checker_tool import data_completeness_checker_tool
from backned.mcp.tools.business_validation_tool import business_validation_tool
from backned.mcp.tools.insight_reporter_tool import insight_reporter_tool

#  RAG tools (implementations in backned/mcp/tools/rag_tools/) 
from backned.mcp.tools.rag_tools.vector_indexer_tool import vector_indexer_tool
from backned.mcp.tools.rag_tools.semantic_retriever_tool import semantic_retriever_tool
from backned.mcp.tools.rag_tools.chunk_ranker_tool import chunk_ranker_tool
from backned.mcp.tools.rag_tools.response_synthesizer_tool import response_synthesizer_tool

logger = logging.getLogger(__name__)

# 
# FastMCP server instance
# 
mcp = FastMCP(
    name="Invoice Auditor",
    instructions=(
        "MCP server for the Invoice Auditor pipeline. "
        "Provides tools to monitor, extract, translate, validate, "
        "cross-check against ERP, generate audit reports, and answer "
        "natural language questions about invoices via RAG."
    ),
)


# 
#  PIPELINE TOOLS
# 

@mcp.tool(
    name="invoice_watcher",
    description=(
        "Monitor the incoming-emails directory for new, unprocessed "
        "invoice emails. Returns a MonitorResult with new_emails, "
        "counts, and any scan errors."
    ),
)
def watch_invoices(scan_directory: str | None = None) -> dict:
    """Invoke the Invoice-Watcher Tool.

    Args:
        scan_directory: Optional path override for the emails directory.
                        Defaults to ``backned/data/incoming/emails/``.

    Returns:
        dict serialised from MonitorResult (new_emails, counts, errors).
    """
    return invoice_watcher_tool.invoke({"scan_directory": scan_directory})


@mcp.tool(
    name="data_harvester",
    description=(
        "Extract structured invoice data from a PDF, DOCX, or image "
        "file using Azure OpenAI GPT-4o vision.  Returns a full "
        "ExtractionResult dict with header fields and line items."
    ),
)
def harvest_invoice(file_path: str, email_id: str = "unknown") -> dict:
    """Invoke the Data Harvester Tool.

    Args:
        file_path: Absolute path to the invoice file (PDF/DOCX/image).
        email_id:  Identifier of the email this invoice belongs to.

    Returns:
        dict serialised from ExtractionResult.
    """
    return data_harvester_tool.invoke({"file_path": file_path, "email_id": email_id})


@mcp.tool(
    name="lang_bridge",
    description=(
        "Translate extracted invoice content from any language into "
        "English using Azure OpenAI GPT-4o.  English invoices are "
        "passed through unchanged.  Returns a TranslationResult dict."
    ),
)
def translate_invoice(extraction_result: dict) -> dict:
    """Invoke the Lang-Bridge Tool.

    Args:
        extraction_result: dict serialised from ExtractionResult
                           (output of the data_harvester tool).

    Returns:
        dict serialised from TranslationResult.
    """
    return lang_bridge_tool.invoke({"extraction_result": extraction_result})


@mcp.tool(
    name="data_completeness_checker",
    description=(
        "Validate a translated invoice for completeness and accuracy "
        "against the business rules defined in ``config/rules.yaml``. "
        "Checks required fields, data types, date formats, currency "
        "codes, and invoice-level / line-item-level maths."
    ),
)
def check_data_completeness(translation_result: dict) -> dict:
    """Invoke the Data Completeness Checker Tool.

    Args:
        translation_result: dict serialised from TranslationResult
                            (output of the lang_bridge tool).

    Returns:
        dict serialised from ValidationResult.
    """
    return data_completeness_checker_tool.invoke(
        {"translation_result": translation_result}
    )


@mcp.tool(
    name="business_validation",
    description=(
        "Cross-verify invoice line items against the Mock ERP (PO "
        "records, vendor master, SKU catalogue).  Detects qty, "
        "unit-price, vendor, and currency discrepancies. "
        "Returns a BusinessValidationResult dict."
    ),
)
def validate_against_erp(
    translation_result: dict,
    validation_result: dict | None = None,
) -> dict:
    """Invoke the Business Validation Tool.

    Args:
        translation_result: dict serialised from TranslationResult.
        validation_result:  Optional ValidationResult dict (pass-through
                            context from the completeness check).

    Returns:
        dict serialised from BusinessValidationResult.
    """
    return business_validation_tool.invoke(
        {
            "translation_result": translation_result,
            "validation_result": validation_result,
        }
    )


@mcp.tool(
    name="insight_reporter",
    description=(
        "Generate a comprehensive HTML audit report for a single "
        "invoice by aggregating results from translation, data "
        "validation, and business validation stages.  Writes the "
        "HTML file to ``outputs/reports/`` and returns an "
        "InvoiceReport dict that includes the file path."
    ),
)
def generate_report(
    translation_result: dict,
    validation_result: dict | None = None,
    business_validation_result: dict | None = None,
) -> dict:
    """Invoke the Insight Reporter Tool.

    Args:
        translation_result:          TranslationResult dict.
        validation_result:           ValidationResult dict (optional).
        business_validation_result:  BusinessValidationResult dict (optional).

    Returns:
        dict serialised from InvoiceReport (includes report_html_path).
    """
    return insight_reporter_tool.invoke(
        {
            "translation_result": translation_result,
            "validation_result": validation_result,
            "business_validation_result": business_validation_result,
        }
    )


# 
#  RAG TOOLS
# 

@mcp.tool(
    name="vector_indexer",
    description=(
        "Index all invoice reports, raw invoices, and ERP mock-data "
        "into the FAISS vector store using Azure OpenAI embeddings. "
        "Only new or changed documents are processed unless "
        "force_reindex=True.  Returns indexing statistics."
    ),
)
def index_documents(force_reindex: bool = False) -> dict:
    """Invoke the Vector Indexer Tool.

    Args:
        force_reindex: If True, rebuilds the entire FAISS index from
                       scratch (ignores previously indexed content).

    Returns:
        dict with keys: chunks_indexed (int), new_chunks (int),
        skipped_sources (list[str]), sources (list[str]),
        index_path (str), status (str).
    """
    return vector_indexer_tool.invoke({"force_reindex": force_reindex})


@mcp.tool(
    name="semantic_retriever",
    description=(
        "Perform a semantic similarity search against the FAISS "
        "invoice index for a natural-language query.  Returns the "
        "top-k most relevant document chunks before re-ranking."
    ),
)
def retrieve_chunks(query: str, top_k: int = 10) -> dict:
    """Invoke the Semantic Retriever Tool.

    Args:
        query: Natural language question from the user.
        top_k: Number of candidate chunks to return before re-ranking.

    Returns:
        dict with keys: query (str), chunks (list of dicts with
        text / source / score), total_indexed (int), status (str).
    """
    return semantic_retriever_tool.invoke({"query": query, "top_k": top_k})


@mcp.tool(
    name="chunk_ranker",
    description=(
        "Re-rank retrieved chunks using keyword boosting, source-type "
        "weighting, and maximum marginal relevance (MMR) to surface "
        "the most relevant context for a given query."
    ),
)
def rank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = 8,
) -> dict:
    """Invoke the Chunk Ranker Tool.

    Args:
        query:  Original user query.
        chunks: List of chunk dicts from the semantic_retriever tool.
        top_n:  Final number of top chunks to keep after re-ranking.

    Returns:
        dict with keys: query (str), ranked_chunks (list[dict]),
        total_input (int), total_output (int), status (str).
    """
    return chunk_ranker_tool.invoke({"query": query, "chunks": chunks, "top_n": top_n})


@mcp.tool(
    name="response_synthesizer",
    description=(
        "Generate a grounded natural-language answer from ranked "
        "evidence chunks and optional conversation history using "
        "Azure OpenAI GPT-4o.  Answers are strictly derived from "
        "the provided context — no hallucinations."
    ),
)
def synthesize_response(
    query: str,
    ranked_chunks: list[dict[str, Any]],
    chat_history: list[dict[str, str]] | None = None,
) -> dict:
    """Invoke the Response Synthesizer Tool.

    Args:
        query:         The user's current natural-language question.
        ranked_chunks: Top-N ranked context chunks from chunk_ranker.
        chat_history:  Prior conversation turns as a list of
                       ``{'role': 'user'|'assistant', 'content': str}``
                       dicts (optional).

    Returns:
        dict with keys: answer (str), sources_used (list[str]),
        context_chunks (int), status (str).
    """
    return response_synthesizer_tool.invoke(
        {
            "query": query,
            "ranked_chunks": ranked_chunks,
            "chat_history": chat_history,
        }
    )


# 
# Entry point  — SSE transport
# 
HOST = "127.0.0.1"
PORT = 8000

if __name__ == "__main__":
    logger.info(
        "Starting Invoice Auditor MCP Server  [SSE]  http://%s:%s/sse",
        HOST, PORT,
    )
    mcp.run(transport="sse", host=HOST, port=PORT)
