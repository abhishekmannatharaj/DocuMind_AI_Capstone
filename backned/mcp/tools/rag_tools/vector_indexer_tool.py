"""Vector Indexer Tool — converts invoice documents, ERP data, and audit reports
into text chunks, embeds them via Azure OpenAI, and stores them in a FAISS index.

Exposes `vector_indexer_tool` as a LangChain @tool for use by the Indexing Agent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_core.tools import tool
from backned.config.settings import (
    AZURE_AI_ENDPOINT,
    API_KEY,
    API_VERSION,
    EMBEDDING_MODEL,
)
from backned.services.llm_provider import (
    get_embedding,
    get_embeddings,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

#  Paths 
# File lives at backned/mcp/tools/rag_tools/ — 4 levels deep from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RAG_DIR      = _PROJECT_ROOT / "backned" / "rag_store"
_INDEX_PATH   = _RAG_DIR / "invoice_index.faiss"
_META_PATH    = _RAG_DIR / "invoice_meta.pkl"
_REPORTS_DIR  = _PROJECT_ROOT / "outputs" / "reports"
_ERP_DIR      = _PROJECT_ROOT / "backned" / "data" / "ERP_mockdata"
_INVOICES_DIR = _PROJECT_ROOT / "backned" / "data" / "incoming" / "invoices"

_EMBEDDING_DIM = EMBEDDING_DIM   # 384 for sentence-transformers / offline fallback
_CHUNK_SIZE    = 800    # target chars per chunk (raised for richer content)
_CHUNK_OVERLAP = 150


#  @tool entry point 

def _file_fingerprint(path: Path) -> str:
    """Return a cheap fingerprint (mtime_ns:size) for a file.

    Used to detect whether a source file has changed since it was last indexed.
    Avoids reading the full file on every run.
    """
    s = path.stat()
    return f"{s.st_mtime_ns}:{s.st_size}"


@tool
def vector_indexer_tool(force_reindex: bool = False) -> dict:
    """Index all invoice reports, raw invoices, and ERP mock data into FAISS.

    Only processes source files that have NOT been indexed before (or whose
    content has changed since the last run).  Pass force_reindex=True to
    rebuild the entire index from scratch.

    Args:
        force_reindex: If True, rebuilds the entire index from scratch.

    Returns:
        dict with keys: chunks_indexed (int), new_chunks (int),
        skipped_sources (list[str]), sources (list[str]),
        index_path (str), status (str).
    """
    _RAG_DIR.mkdir(parents=True, exist_ok=True)

    #  Load existing metadata 
    existing_hashes: set[str] = set()
    all_chunks: list[dict] = []
    stored_vectors: np.ndarray = np.zeros((0, _EMBEDDING_DIM), dtype=np.float32)
    indexed_sources: dict[str, str] = {}   # label → fingerprint

    if _META_PATH.exists() and not force_reindex:
        with open(_META_PATH, "rb") as fh:
            stored = pickle.load(fh)
        all_chunks      = stored.get("chunks", [])
        existing_hashes = {c["hash"] for c in all_chunks}
        indexed_sources = stored.get("indexed_sources", {})
        raw_vecs        = stored.get("vectors", [])
        if raw_vecs:
            arr = np.array(raw_vecs, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == _EMBEDDING_DIM:
                stored_vectors = arr
            else:
                logger.info("Existing index dimension %s != %s; reindexing with local embedding", arr.shape, _EMBEDDING_DIM)
                all_chunks = []
                existing_hashes = set()
                indexed_sources = {}
        logger.info(
            "Loaded %d existing chunks from %d indexed source(s)",
            len(all_chunks), len(indexed_sources),
        )

    #  Discover source files 
    # Maps source_label → Path for every candidate document
    candidate_sources: dict[str, Path] = {}

    if _REPORTS_DIR.exists():
        for f in _REPORTS_DIR.glob("*.html"):
            candidate_sources[f"report:{f.name}"] = f

    if _ERP_DIR.exists():
        for f in _ERP_DIR.glob("*.json"):
            candidate_sources[f"erp:{f.name}"] = f

    #  Always include the synthetic aggregate summary 
    # Its «file» is a sentinel; fingerprint is built from all report fps.
    candidate_sources["summary:all_invoices"] = None  # type: ignore[assignment]

    #  Decide which sources need indexing 
    new_docs: list[tuple[str, str]] = []       # (text, source_label) to index
    skipped_sources: list[str]      = []       # already up-to-date
    pending_fps: dict[str, str]     = {}       # label → fingerprint for this run

    # Build aggregate-summary fingerprint from all current report fingerprints
    _report_fps = {
        lbl: _file_fingerprint(p)
        for lbl, p in candidate_sources.items()
        if p is not None and lbl.startswith("report:")
    }
    _summary_fp = hashlib.sha256(
        "|".join(sorted(f"{k}={v}" for k, v in _report_fps.items())).encode()
    ).hexdigest()[:16]

    for label, path in candidate_sources.items():
        # Aggregate summary: regenerate when any report changed
        if label == "summary:all_invoices":
            if not force_reindex and indexed_sources.get(label) == _summary_fp:
                skipped_sources.append(label)
                continue
            text = _build_aggregate_summary(_REPORTS_DIR)
            new_docs.append((text, label))
            pending_fps[label] = _summary_fp
            continue

        fp = _file_fingerprint(path)
        if not force_reindex and indexed_sources.get(label) == fp:
            skipped_sources.append(label)
            continue
        # New file or changed file — queue for indexing
        text = (
            _report_html_to_text(path)
            if label.startswith("report:")
            else _erp_json_to_text(path)
        )
        new_docs.append((text, label))
        pending_fps[label] = fp

    logger.info(
        "%d new/changed source(s) to index | %d already up-to-date",
        len(new_docs), len(skipped_sources),
    )

    #  Early-exit when nothing changed 
    if not new_docs and all_chunks:
        return {
            "chunks_indexed":  len(all_chunks),
            "new_chunks":      0,
            "skipped_sources": skipped_sources,
            "sources":         list({c["source"] for c in all_chunks}),
            "index_path":      str(_INDEX_PATH),
            "status":          "up_to_date",
        }

    #  Chunk new documents 
    # Sources that must NEVER be split — stored as one atomic chunk each so
    # the LLM always receives the full context in a single retrieval hit:
    #   • summary:all_invoices  — aggregate cross-invoice reference doc
    #   • report:*              — individual audit reports (~1700 chars each);
    #                             splitting them separates header/errors/detail
    _NO_SPLIT_PREFIXES = ("summary:", "report:")

    new_chunks: list[dict] = []
    for text, source in new_docs:
        if source.startswith(_NO_SPLIT_PREFIXES):
            chunk_hash = _sha(text)
            if chunk_hash not in existing_hashes:
                new_chunks.append({"text": text, "source": source, "hash": chunk_hash})
                existing_hashes.add(chunk_hash)
            continue

        for chunk_text in _split_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP):
            chunk_hash = _sha(chunk_text)
            if chunk_hash not in existing_hashes:
                new_chunks.append({
                    "text":   chunk_text,
                    "source": source,
                    "hash":   chunk_hash,
                })
                existing_hashes.add(chunk_hash)

    logger.info("%d new chunk(s) to embed", len(new_chunks))

    # If the files were scanned but contained only already-known text (edge case),
    # still record their fingerprints so they're skipped on the next run.
    if not new_chunks:
        indexed_sources.update(pending_fps)
        meta_chunks = [{k: v for k, v in c.items() if k != "vector"} for c in all_chunks]
        with open(_META_PATH, "wb") as fh:
            pickle.dump({
                "chunks":          meta_chunks,
                "vectors":         stored_vectors.tolist(),
                "indexed_sources": indexed_sources,
            }, fh)
        return {
            "chunks_indexed":  len(all_chunks),
            "new_chunks":      0,
            "skipped_sources": skipped_sources,
            "sources":         list({c["source"] for c in all_chunks}),
            "index_path":      str(_INDEX_PATH),
            "status":          "up_to_date",
        }

    #  Embed new chunks 
    new_vectors = np.array(
        _embed_batch([c["text"] for c in new_chunks]), dtype=np.float32
    )

    #  Merge with existing vectors 
    if stored_vectors.shape[0] > 0:
        all_vectors = np.vstack([stored_vectors, new_vectors])
    else:
        all_vectors = new_vectors

    all_chunks.extend(new_chunks)

    #  Build / update FAISS index 
    # Normalise for cosine similarity (IndexFlatIP)
    faiss.normalize_L2(all_vectors)
    index = faiss.IndexFlatIP(_EMBEDDING_DIM)
    index.add(all_vectors)
    faiss.write_index(index, str(_INDEX_PATH))

    #  Persist metadata 
    indexed_sources.update(pending_fps)
    meta_chunks = [{k: v for k, v in c.items() if k != "vector"} for c in all_chunks]
    with open(_META_PATH, "wb") as fh:
        pickle.dump({
            "chunks":          meta_chunks,
            "vectors":         all_vectors.tolist(),
            "indexed_sources": indexed_sources,
        }, fh)

    sources = list({c["source"] for c in all_chunks})
    logger.info(
        "Index built: %d total chunks (%d new) across %d source(s)",
        len(all_chunks), len(new_chunks), len(sources),
    )

    return {
        "chunks_indexed":  len(all_chunks),
        "new_chunks":      len(new_chunks),
        "skipped_sources": skipped_sources,
        "sources":         sources,
        "index_path":      str(_INDEX_PATH),
        "status":          "indexed",
    }


#  Public helpers (used by other tools) 

def load_index() -> tuple[faiss.Index, list[dict], np.ndarray] | None:
    """Load the FAISS index + metadata from disk.  Returns None if not built or dimension mismatch."""
    if not _INDEX_PATH.exists() or not _META_PATH.exists():
        return None
    try:
        index = faiss.read_index(str(_INDEX_PATH))
        if index.d != _EMBEDDING_DIM:
            return None
        with open(_META_PATH, "rb") as fh:
            stored = pickle.load(fh)
        chunks  = stored["chunks"]
        vectors = np.array(stored["vectors"], dtype=np.float32)
        if vectors.shape[1] != _EMBEDDING_DIM:
            return None
        return index, chunks, vectors
    except Exception:
        return None


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string and return an L2-normalised vector."""
    vec = get_embedding(text)
    vec = vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    return vec


#  Private helpers 

def _embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts using local sentence-transformers or offline fallback."""
    return get_embeddings(texts)


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Chunk text with overlap, preferring sentence boundaries."""
    text   = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        next_start = end - overlap
        if next_start <= start:
            break   # guard against infinite loop near end of text
        start = next_start
    return [c for c in chunks if len(c) > 30]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _extract_field(html: str, label: str) -> str:
    """Extract a value from the invoice HTML report table.

    Handles two layouts:
      1. <div class="fl">LABEL</div><div class="fv ...">VALUE</div>
      2. <span>LABEL</span><span class="fv-lg">VALUE</span>
    """
    # Layout 1: fl/fv div pair
    m = re.search(
        re.escape(label) + r'</div>\s*<div[^>]*class="fv[^"]*"[^>]*>([^<]+)',
        html, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    # Layout 2: span pair (used for Total Amount)
    m = re.search(
        re.escape(label) + r'</span>\s*<span[^>]*class="fv[^"]*"[^>]*>([^<]+)',
        html, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    return "N/A"


def _report_html_to_text(path: Path) -> str:
    """Extract structured text from an audit HTML report.

    Strips CSS/JS entirely and produces a rich, structured plain-text document
    containing all invoice fields, verdict, discrepancies and line items.
    """
    html = path.read_text(encoding="utf-8", errors="replace")

    #  Extract key fields 
    inv_id   = re.search(r'topbar-sub">(.*?)&ensp;', html)
    verdict  = re.search(r'class="verdict-label">(.*?)</span>', html)
    conf     = re.search(r'Auto-[Aa]pprove confidence:\s*<strong>(\d+)%', html)

    inv_no    = _extract_field(html, "Invoice No.")
    inv_date  = _extract_field(html, "Invoice Date")
    po_no     = _extract_field(html, "PO Number")
    vendor_id = _extract_field(html, "Vendor ID")
    vendor    = _extract_field(html, "Vendor Name")
    total     = _extract_field(html, "Total Amount")
    currency  = _extract_field(html, "Currency")

    def gm(m): return m.group(1).strip() if m else "N/A"

    verdict_val = gm(verdict)
    if "auto" in verdict_val.lower():
        verdict_norm = "Auto-Approved"
    elif "manual" in verdict_val.lower():
        verdict_norm = "Manual Review"
    elif "flag" in verdict_val.lower():
        verdict_norm = "Flagged"
    elif "reject" in verdict_val.lower():
        verdict_norm = "Rejected"
    else:
        verdict_norm = verdict_val

    header_lines = [
        f"=== AUDIT REPORT: {path.stem} ===",
        f"Invoice ID      : {gm(inv_id)}",
        f"Invoice Number  : {inv_no}",
        f"Invoice Date    : {inv_date}",
        f"PO Number       : {po_no}",
        f"Vendor ID       : {vendor_id}",
        f"Vendor Name     : {vendor}",
        f"Currency        : {currency}",
        f"Total Amount    : {total}",
        f"Verdict         : {verdict_norm}",
        f"Confidence Score: {gm(conf)}%",
        "",
    ]

    #  Extract body text only (skip CSS/JS) 
    body_start = max(html.rfind("</style>"), html.rfind("</head>"))
    body_html  = html[body_start:] if body_start != -1 else html
    body_html  = re.sub(r"<script[^>]*>.*?</script>", " ", body_html, flags=re.DOTALL | re.IGNORECASE)
    body_html  = re.sub(r"<style[^>]*>.*?</style>",  " ", body_html, flags=re.DOTALL | re.IGNORECASE)

    clean = re.sub(r"<[^>]+>", " ", body_html)
    clean = re.sub(r"&[a-zA-Z#0-9]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return "\n".join(header_lines) + clean


def _build_aggregate_summary(reports_dir: Path) -> str:
    """Build a synthetic aggregate document listing ALL invoices and verdicts."""
    rows: list[dict] = []
    for path in sorted(reports_dir.glob("*.html")):
        html = path.read_text(encoding="utf-8", errors="replace")
        inv_id  = re.search(r'topbar-sub">(.*?)&ensp;', html)
        verdict = re.search(r'class="verdict-label">(.*?)</span>', html)
        conf    = re.search(r'Auto-[Aa]pprove confidence:\s*<strong>(\d+)%', html)

        inv_no  = _extract_field(html, "Invoice No.")
        vendor  = _extract_field(html, "Vendor Name")
        total   = _extract_field(html, "Total Amount")
        po_no   = _extract_field(html, "PO Number")

        def gm(m): return m.group(1).strip() if m else "N/A"

        verdict_val = gm(verdict)
        if "auto" in verdict_val.lower():
            verdict_norm = "Auto-Approved"
        elif "manual" in verdict_val.lower():
            verdict_norm = "Manual Review"
        elif "flag" in verdict_val.lower():
            verdict_norm = "Flagged"
        elif "reject" in verdict_val.lower():
            verdict_norm = "Rejected"
        else:
            verdict_norm = verdict_val

        rows.append({
            "file":    path.stem,
            "inv_id":  gm(inv_id),
            "inv_no":  inv_no,
            "vendor":  vendor,
            "total":   total,
            "po_no":   po_no,
            "verdict": verdict_norm,
            "conf":    gm(conf),
        })

    auto_approved  = [r for r in rows if r["verdict"] == "Auto-Approved"]
    manual_review  = [r for r in rows if r["verdict"] == "Manual Review"]
    flagged        = [r for r in rows if r["verdict"] == "Flagged"]
    rejected       = [r for r in rows if r["verdict"] == "Rejected"]

    lines = [
        "=== AGGREGATE INVOICE AUDIT SUMMARY ===",
        f"Total Invoices Processed : {len(rows)}",
        f"Auto-Approved            : {len(auto_approved)}",
        f"Manual Review            : {len(manual_review)}",
        f"Flagged                  : {len(flagged)}",
        f"Rejected                 : {len(rejected)}",
        "",
        "--- All Invoices and Their Verdicts ---",
    ]
    for r in rows:
        lines.append(
            f"Invoice: {r['inv_id']} | Number: {r['inv_no']} | Vendor: {r['vendor']} "
            f"| PO: {r['po_no']} | Total: {r['total']} "
            f"| Verdict: {r['verdict']} | Confidence: {r['conf']}%"
        )

    lines += [
        "",
        f"--- Auto-Approved Invoices ({len(auto_approved)}) ---",
    ]
    for r in auto_approved:
        lines.append(f"  {r['inv_id']} ({r['inv_no']}) — {r['vendor']} — {r['total']} — {r['conf']}% confidence")

    lines += [
        "",
        f"--- Manual Review Invoices ({len(manual_review)}) ---",
    ]
    for r in manual_review:
        lines.append(f"  {r['inv_id']} ({r['inv_no']}) — {r['vendor']} — {r['total']} — {r['conf']}% confidence")

    if flagged:
        lines += ["", f"--- Flagged Invoices ({len(flagged)}) ---"]
        for r in flagged:
            lines.append(f"  {r['inv_id']} ({r['inv_no']}) — {r['vendor']} — {r['total']}")

    if rejected:
        lines += ["", f"--- Rejected Invoices ({len(rejected)}) ---"]
        for r in rejected:
            lines.append(f"  {r['inv_id']} ({r['inv_no']}) — {r['vendor']} — {r['total']}")

    return "\n".join(lines)


def _erp_json_to_text(path: Path) -> str:
    """Convert ERP JSON file to a richly-structured text for indexing.

    Prepends a flat one-line summary of all record IDs so that aggregate
    queries like 'show all vendor IDs' are answered from the very first
    chunk, even if the detailed records get split later.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    lines = [f"ERP DATA FILE: {path.name}"]
    records = data if isinstance(data, list) else [data]
    name_lower = path.name.lower()

    #  Flat summary line (all IDs / keys in one scannable line) 
    if "vendor" in name_lower:
        summary_parts = [
            f"{r.get('vendor_id','?')} ({r.get('vendor_name','?')}, {r.get('country','?')}, {r.get('currency','?')})"
            for r in records
        ]
        lines.append("ALL VENDOR IDs: " + " | ".join(summary_parts))
        lines.append(f"Total vendors in system: {len(records)}")
    elif "sku" in name_lower or "master" in name_lower:
        summary_parts = [
            f"{r.get('item_code','?')} ({r.get('category','?')}, {r.get('uom','?')})"
            for r in records
        ]
        lines.append("ALL SKU / ITEM CODES: " + " | ".join(summary_parts))
        lines.append(f"Total SKUs in system: {len(records)}")
    elif "po" in name_lower or "purchase" in name_lower or "order" in name_lower:
        summary_parts = [
            f"{r.get('po_number','?')} → vendor {r.get('vendor_id','?')}"
            for r in records
        ]
        lines.append("ALL PO NUMBERS: " + " | ".join(summary_parts))
        lines.append(f"Total POs in system: {len(records)}")

    lines.append("")

    #  Detailed records (human-readable, not raw JSON) 
    for r in records:
        if "vendor" in name_lower:
            lines.append(
                f"Vendor ID: {r.get('vendor_id')}  |  Name: {r.get('vendor_name')}  "
                f"|  Country: {r.get('country')}  |  Currency: {r.get('currency')}"
            )
        elif "sku" in name_lower or "master" in name_lower:
            lines.append(
                f"Item Code: {r.get('item_code')}  |  Category: {r.get('category')}  "
                f"|  UoM: {r.get('uom')}  |  GST Rate: {r.get('gst_rate')}%"
            )
        elif "po" in name_lower or "purchase" in name_lower or "order" in name_lower:
            items_str = ", ".join(
                f"{li.get('item_code')} x{li.get('qty')} @ {li.get('unit_price')} {li.get('currency','')}"
                for li in r.get("line_items", [])
            )
            lines.append(
                f"PO: {r.get('po_number')}  |  Vendor: {r.get('vendor_id')}  "
                f"|  Items: {items_str}"
            )
        else:
            lines.append(json.dumps(r, ensure_ascii=False))

    return "\n".join(lines)
