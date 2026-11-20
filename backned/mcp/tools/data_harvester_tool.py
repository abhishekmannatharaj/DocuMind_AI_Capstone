"""Data Harvester Tool — extracts structured invoice data from PDF, DOCX, and images via Azure OpenAI GPT-4o.

Exposes a LangChain @tool for use by the Extractor Agent.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

from backned.models import (
    ExtractedInvoice,
    ExtractedLineItem,
    ExtractionResult,
)
from backned.config.settings import (
    AZURE_AI_ENDPOINT,
    API_KEY,
    API_VERSION,
    DEPLOYMENT_NAME,
)
from backned.services.llm_provider import (
    call_chat_llm,
    deterministic_extract_invoice,
)

logger = logging.getLogger(__name__)

#  Supported extensions 
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx", ".doc"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

#  Module-level singleton variable 
_client: AzureOpenAI | None = None


# 
# Data Harvester Tool  (LangChain @tool)
# 
@tool
def data_harvester_tool(
    file_path: str,
    email_id: str = "unknown",
) -> dict:
    """Data Harvester Tool — Extracts structured invoice data from
    PDF, DOCX, and image files using Azure OpenAI GPT-4o.

    1. Detects file format from extension.
    2. For PDF/DOCX: extracts raw text offline, then sends to GPT-4o
       for structured JSON parsing.
    3. For images: sends directly to GPT-4o Vision for both text
       extraction and structured parsing in one call.
    4. Returns an ExtractionResult dict with full structured data.

    Args:
        file_path: Absolute path to the invoice file.
        email_id:  Email ID this invoice belongs to.

    Returns:
        dict serialised from ExtractionResult.
    """
    fp = Path(file_path)

    #  Guard: file must exist 
    if not fp.exists():
        return ExtractionResult(
            email_id=email_id,
            filename=fp.name,
            file_path=str(fp),
            file_format="unknown",
            extraction_errors=[f"File not found: {fp}"],
        ).model_dump()

    # Multi-provider setup (Groq / Gemini / Ollama / Azure / Offline Fallback)
    has_azure = bool(API_KEY and AZURE_AI_ENDPOINT and DEPLOYMENT_NAME)
    if not has_azure:
        logger.info("Azure OpenAI not configured; using local/configured LLM provider or offline fallback")

    ext = fp.suffix.lower()
    file_format = _detect_format(ext)

    raw_text = ""
    parsed_data: dict = {}
    errors: list[str] = []

    try:
        if file_format == "pdf":
            logger.info("Extracting text from PDF: %s", fp.name)
            raw_text = _extract_text_from_pdf(fp)
            if raw_text.strip():
                logger.info("Parsing with GPT-4o (%d chars) ...", len(raw_text))
                parsed_data = _parse_with_gpt(raw_text)
            else:
                errors.append("No text extracted from PDF")

        elif file_format == "docx":
            logger.info("Extracting text from DOCX: %s", fp.name)
            raw_text = _extract_text_from_docx(fp)
            if raw_text.strip():
                logger.info("Parsing with GPT-4o (%d chars) ...", len(raw_text))
                parsed_data = _parse_with_gpt(raw_text)
            else:
                errors.append("No text extracted from DOCX")

        elif file_format == "image":
            logger.info("Sending image to GPT-4o Vision: %s", fp.name)
            raw_text, parsed_data = _parse_image_with_gpt_vision(fp)

        else:
            errors.append(f"Unsupported format: {ext}")

    except json.JSONDecodeError as exc:
        errors.append(f"GPT-4o returned invalid JSON: {exc}")
        logger.error("JSON parse error for %s: %s", fp.name, exc)
    except Exception as exc:
        errors.append(f"Extraction failed for {fp.name}: {exc}")
        logger.error("Extraction error for %s: %s", fp.name, exc)

    #  Build structured invoice 
    invoice: ExtractedInvoice | None = None
    if parsed_data:
        try:
            invoice = _dict_to_extracted_invoice(parsed_data)
        except Exception as exc:
            errors.append(f"Failed to build invoice model: {exc}")
            logger.error("Model build error: %s", exc)

    #  Validate currency is actually present in raw text (or implied by Indian GSTIN)
    if invoice and invoice.additional_fields.get("gstin"):
        invoice.currency = "INR"
    elif invoice and invoice.currency and raw_text:
        invoice.currency = _validate_currency_in_text(
            invoice.currency, raw_text, fp.name
        )

    #  Resolve vendor_id from vendor_name if missing 
    if invoice and not invoice.vendor_id and invoice.vendor_name:
        resolved = _resolve_vendor_id(invoice.vendor_name)
        if resolved:
            invoice.vendor_id = resolved["vendor_id"]
            logger.info(
                "Resolved vendor_id: '%s' → %s (score: %.0f%%)",
                invoice.vendor_name, resolved["vendor_id"],
                resolved["score"] * 100,
            )

    #  Resolve missing item_code from description 
    if invoice and invoice.line_items:
        _resolve_item_codes(invoice)

    result = ExtractionResult(
        email_id=email_id,
        filename=fp.name,
        file_path=str(fp),
        file_format=file_format,
        raw_text=raw_text,
        extracted_invoice=invoice,
        extraction_errors=errors,
    )

    if invoice:
        n_items = len(invoice.line_items)
        logger.info(
            "%s → Invoice#: %s | PO: %s | Vendor: %s | Items: %d | Total: %s %s",
            fp.name,
            invoice.invoice_no or "N/A",
            invoice.po_number or "N/A",
            invoice.vendor_id or invoice.vendor_name or "N/A",
            n_items,
            invoice.currency or "?",
            invoice.total_amount,
        )

    return result.model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


# 
# Format detection
# 

def _detect_format(ext: str) -> str:
    ext = ext.lower()
    if ext in PDF_EXTS:
        return "pdf"
    if ext in DOCX_EXTS:
        return "docx"
    if ext in IMAGE_EXTS:
        return "image"
    return "unknown"


# 
# Azure OpenAI client (lazy singleton)
# 

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_AI_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
    return _client


# 
# Shared GPT-4o extraction prompt
# 
_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert invoice data extractor.  Your job is to extract
EVERY piece of information from the invoice and return it as JSON.

Return ONLY valid JSON — no markdown fences, no commentary.

The JSON schema you MUST follow:
{
  "invoice_no": "<string or null>",
  "invoice_date": "<string in any date format found, or null>",
  "po_number": "<Purchase Order reference like PO-1001, or null>",
  "vendor_id": "<Vendor ID like VEND-001, or null>",
  "vendor_name": "<string or null>",
  "currency": "<ISO 4217 code like USD, EUR, INR, GBP — or null>",
  "total_amount": <number or null>,
  "tax_amount": <number or null>,
  "subtotal": <number or null>,
  "line_items": [
    {
      "item_code": "<SKU code like SKU-001, or null>",
      "description": "<item description in ORIGINAL language — do NOT translate>",
      "qty": <number or null>,
      "unit_price": <number or null>,
      "total": <number or null>
    }
  ],
  "detected_language": "<ISO 639-1 code of the document language, e.g. en, de, es>",
  "additional_fields": { "<any other key-value pairs you find>" }
}

Rules:
- Extract ALL line items, even if the table has many rows.
- Preserve the original language of descriptions — do NOT translate.
- For currency, convert symbols to ISO codes ($ → USD, € → EUR, ₹ → INR, £ → GBP).
  CRITICAL: Set currency to null if NO currency symbol or ISO code is EXPLICITLY AND VISIBLY
  present in the document. NEVER guess, infer, or assume a currency from the country,
  company name, or amount magnitude. Only set it if you can directly see it in the text.
- If a field is not present in the document, set it to null.
- Numbers must be JSON numbers, not strings.
- Do NOT wrap the JSON in ```json``` markers.
"""


# 
# Raw text extractors (fast, offline)
# 

def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract raw text from a PDF using pdfplumber, with RapidOCR fallback for scanned PDFs."""
    import pdfplumber

    parts: list[str] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                parts.append(text)

                # Also stringify any tables
                tables = page.extract_tables() or []
                for tbl in tables:
                    for row in tbl:
                        if row:
                            cells = [c if c else "" for c in row]
                            parts.append(" | ".join(cells))
    except Exception as exc:
        logger.warning("pdfplumber extraction error on %s: %s", file_path.name, exc)

    extracted = "\n".join(parts).strip()
    if len(extracted) >= 20:
        return extracted

    # Scanned PDF fallback: render pages to images and run RapidOCR
    logger.info("PDF %s contains little or no digital text (%d chars). Falling back to RapidOCR...", file_path.name, len(extracted))
    return _ocr_pdf_with_rapidocr(file_path)


def _ocr_pdf_with_rapidocr(file_path: Path) -> str:
    """Run RapidOCR on all pages of a scanned PDF via PyMuPDF (fitz)."""
    try:
        import io
        import fitz
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        doc = fitz.open(str(file_path))
        lines: list[str] = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))
            result, _ = ocr(img)
            if result:
                for box in result:
                    if len(box) > 1 and box[1]:
                        lines.append(str(box[1]))
        ocr_text = "\n".join(lines).strip()
        logger.info("RapidOCR extracted %d lines from scanned PDF %s", len(lines), file_path.name)
        return ocr_text
    except Exception as e:
        logger.warning("RapidOCR failed on %s: %s", file_path.name, e)
        return ""


def _extract_text_from_docx(file_path: Path) -> str:
    """Extract raw text from a DOCX file."""
    from docx import Document

    doc = Document(str(file_path))
    parts: list[str] = []

    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cells))

    return "\n".join(parts)


# 
# GPT-4o parsing: text → structured JSON
# 

def _parse_with_gpt(raw_text: str, filename: str = "") -> dict:
    """Extract structured invoice JSON via configured LLM (Groq/Gemini/Ollama/Azure)
    or robust deterministic fallback."""
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract all invoice data from the following text and "
                "return the JSON:\n\n"
                f"{raw_text}"
            ),
        },
    ]

    llm_resp = call_chat_llm(messages, max_tokens=4096, temperature=0.0)
    if llm_resp:
        content = llm_resp.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()
        try:
            parsed = json.loads(content)
            # Ensure confidence scores are attached
            det = deterministic_extract_invoice(raw_text, filename)
            if "confidence_scores" not in parsed:
                parsed["confidence_scores"] = det.get("confidence_scores", {})
            return parsed
        except Exception as e:
            logger.warning("LLM returned non-JSON (%s), using deterministic extraction", e)

    # Deterministic fallback
    return deterministic_extract_invoice(raw_text, filename)


# 
# Image OCR: pytesseract / Pillow / LLM fallback
# 

def _parse_image_with_gpt_vision(file_path: Path) -> tuple[str, dict]:
    """Extract text from invoice images via RapidOCR / pytesseract and parse structured data."""
    raw_text = ""
    try:
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        img = Image.open(file_path)
        result, _ = ocr(img)
        if result:
            raw_text = "\n".join([str(box[1]) for box in result if len(box) > 1 and box[1]])
            logger.info("RapidOCR extracted %d lines from image %s", len(result), file_path.name)
    except Exception as exc:
        logger.info("RapidOCR image extraction fallback: %s", exc)

    if not raw_text.strip():
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(file_path)
            raw_text = pytesseract.image_to_string(img)
        except Exception as exc:
            logger.info("pytesseract OCR skipped or unavailable: %s", exc)

    if not raw_text.strip():
        raw_text = f"Scanned Invoice Image: {file_path.name}"

    parsed = _parse_with_gpt(raw_text, filename=file_path.name)
    return raw_text, parsed


# 
# JSON dict → Pydantic models
# 

def _dict_to_extracted_invoice(data: dict) -> ExtractedInvoice:
    """Convert the GPT-returned dict into an ExtractedInvoice model."""
    line_items = []
    for li in data.get("line_items", []):
        qty_val = _safe_float(li.get("qty") or li.get("quantity") or 1.0)
        price_val = _safe_float(li.get("unit_price") or li.get("price") or 0.0)
        tot_val = _safe_float(li.get("total") or li.get("amount") or (qty_val * price_val))
        line_items.append(ExtractedLineItem(
            item_code=li.get("item_code") or "SKU-GEN",
            description=li.get("description") or li.get("item") or "Goods/Services",
            qty=qty_val,
            unit_price=price_val,
            total=tot_val,
        ))

    add_fields = dict(data.get("additional_fields", {}))
    if "confidence_scores" in data:
        add_fields["confidence_scores"] = data["confidence_scores"]

    inv_no = data.get("invoice_no") or add_fields.get("bill_no") or add_fields.get("invoice_number")
    vend_name = data.get("vendor_name") or add_fields.get("vendor_name")
    gstin = add_fields.get("gstin") or add_fields.get("gst_in") or data.get("gstin")
    if gstin:
        add_fields["gstin"] = str(gstin).strip()

    return ExtractedInvoice(
        invoice_no=inv_no,
        invoice_date=data.get("invoice_date"),
        po_number=data.get("po_number"),
        vendor_id=data.get("vendor_id"),
        vendor_name=vend_name,
        currency=data.get("currency") or ("INR" if gstin else None),
        total_amount=_safe_float(data.get("total_amount")),
        tax_amount=_safe_float(data.get("tax_amount")),
        subtotal=_safe_float(data.get("subtotal") or add_fields.get("sub_total")),
        line_items=line_items,
        detected_language=data.get("detected_language") or "en",
        additional_fields=add_fields,
    )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ISO code → currency symbol(s) used for presence check
_CURRENCY_EVIDENCE: dict[str, list[str]] = {
    "USD": ["USD", "$"],
    "EUR": ["EUR", "€"],
    "INR": ["INR", "₹"],
    "GBP": ["GBP", "£"],
    "JPY": ["JPY", "¥"],
    "AUD": ["AUD"],
    "CAD": ["CAD"],
    "CHF": ["CHF"],
    "CNY": ["CNY", "¥"],
    "SGD": ["SGD"],
    "AED": ["AED"],
}


def _validate_currency_in_text(
    currency: str | None, raw_text: str, filename: str = ""
) -> str | None:
    """Return currency only if its code or symbol is explicitly in the raw text.

    Prevents LLM from hallucinating a currency when none is visible in the doc.
    """
    if not currency or not raw_text:
        return currency

    cu = currency.strip().upper()
    evidence_tokens = _CURRENCY_EVIDENCE.get(cu, [cu])

    # Case-insensitive check for ISO code; exact match for symbols
    text_upper = raw_text.upper()
    for token in evidence_tokens:
        if token.isalpha():
            # ISO code — check as standalone word (avoid substring matches)
            if re.search(rf'\b{re.escape(token)}\b', text_upper):
                return currency
        else:
            # Symbol — direct substring check
            if token in raw_text:
                return currency

    logger.warning(
        "[%s] Currency '%s' not found in document text — setting to null "
        "(LLM likely hallucinated it)",
        filename, currency,
    )
    return None


# 
# Vendor ID resolution (vendor_name → vendor_id)
# 
_VENDOR_MATCH_THRESHOLD = 0.60  # minimum similarity to accept a match

# Cache the vendor list so we only read the file once
_vendor_cache: list[dict] | None = None


def _load_vendors() -> list[dict]:
    """Load vendor master from ERP mock data (cached)."""
    global _vendor_cache
    if _vendor_cache is not None:
        return _vendor_cache

    project_root = Path(__file__).resolve().parents[2]  # backned/
    fp = project_root / "data" / "ERP_mockdata" / "vendors.json"
    if not fp.exists():
        logger.warning("Vendor master not found at %s", fp)
        _vendor_cache = []
        return _vendor_cache

    with open(fp, "r", encoding="utf-8") as fh:
        _vendor_cache = json.load(fh)
    logger.info("Loaded %d vendors from %s", len(_vendor_cache), fp.name)
    return _vendor_cache


def _resolve_vendor_id(vendor_name: str) -> dict | None:
    """Fuzzy-match an extracted vendor_name against the vendor master
    and return ``{"vendor_id": ..., "vendor_name": ..., "score": ...}``
    or ``None`` if no match is above the threshold.

    Matching strategy (in priority order):
    1. **Exact match** (case-insensitive)
    2. **Substring containment** (either direction)
    3. **SequenceMatcher fuzzy** score ≥ threshold
    """
    vendors = _load_vendors()
    if not vendors:
        return None

    name_lower = vendor_name.strip().lower()

    best_match: dict | None = None
    best_score: float = 0.0

    for v in vendors:
        master_name = (v.get("vendor_name") or "").strip().lower()
        if not master_name:
            continue

        # 1. Exact match
        if name_lower == master_name:
            return {
                "vendor_id": v["vendor_id"],
                "vendor_name": v["vendor_name"],
                "score": 1.0,
            }

        # 2. Substring containment
        if name_lower in master_name or master_name in name_lower:
            score = 0.90
            if score > best_score:
                best_score = score
                best_match = v
            continue

        # 3. Fuzzy match
        score = SequenceMatcher(None, name_lower, master_name).ratio()
        if score > best_score:
            best_score = score
            best_match = v

    if best_match and best_score >= _VENDOR_MATCH_THRESHOLD:
        return {
            "vendor_id": best_match["vendor_id"],
            "vendor_name": best_match["vendor_name"],
            "score": round(best_score, 2),
        }

    logger.warning(
        "No vendor match for '%s' (best: %.0f%%)",
        vendor_name, best_score * 100,
    )
    return None


# 
# Item-code resolution (description → item_code via PO Records)
# 
_ITEM_CODE_MATCH_THRESHOLD = 0.60

# Cache: {description_lower: item_code}
_po_desc_index: dict[str, str] | None = None


def _load_po_description_index() -> dict[str, str]:
    """Build a {description_lower: item_code} lookup from all PO records.
    Cached after first load."""
    global _po_desc_index
    if _po_desc_index is not None:
        return _po_desc_index

    project_root = Path(__file__).resolve().parents[2]  # backned/
    fp = project_root / "data" / "ERP_mockdata" / "PO Records.json"
    if not fp.exists():
        logger.warning("PO Records not found at %s", fp)
        _po_desc_index = {}
        return _po_desc_index

    with open(fp, "r", encoding="utf-8") as fh:
        records = json.load(fh)

    index: dict[str, str] = {}
    for po in records:
        for li in po.get("line_items", []):
            desc = (li.get("description") or "").strip().lower()
            code = li.get("item_code", "")
            if desc and code:
                index[desc] = code
    _po_desc_index = index
    logger.info("Loaded %d item descriptions from PO Records", len(index))
    return _po_desc_index


def _match_item_code(description: str) -> dict | None:
    """Fuzzy-match a line-item description against PO records
    and return ``{"item_code": ..., "matched_desc": ..., "score": ...}``
    or ``None`` if no match above threshold.

    Matching strategy:
    1. Exact match (case-insensitive)
    2. Substring containment
    3. SequenceMatcher fuzzy ≥ threshold
    """
    index = _load_po_description_index()
    if not index:
        return None

    desc_lower = description.strip().lower()

    best_code: str | None = None
    best_desc: str = ""
    best_score: float = 0.0

    for master_desc, item_code in index.items():
        # 1. Exact match
        if desc_lower == master_desc:
            return {
                "item_code": item_code,
                "matched_desc": master_desc,
                "score": 1.0,
            }

        # 2. Substring containment
        if desc_lower in master_desc or master_desc in desc_lower:
            score = 0.90
            if score > best_score:
                best_score = score
                best_code = item_code
                best_desc = master_desc
            continue

        # 3. Fuzzy match
        score = SequenceMatcher(None, desc_lower, master_desc).ratio()
        if score > best_score:
            best_score = score
            best_code = item_code
            best_desc = master_desc

    if best_code and best_score >= _ITEM_CODE_MATCH_THRESHOLD:
        return {
            "item_code": best_code,
            "matched_desc": best_desc,
            "score": round(best_score, 2),
        }

    return None


def _resolve_item_codes(invoice: ExtractedInvoice) -> None:
    """For each line item missing item_code, try to resolve it
    by fuzzy-matching the description against PO records."""
    for idx, li in enumerate(invoice.line_items):
        if li.item_code:
            continue  # already has item_code
        if not li.description:
            continue  # no description to match against

        match = _match_item_code(li.description)
        if match:
            li.item_code = match["item_code"]
            logger.info(
                "Line %d: resolved item_code '%s' → %s (score: %.0f%%, "
                "matched: '%s')",
                idx + 1, li.description, match["item_code"],
                match["score"] * 100, match["matched_desc"],
            )
        else:
            logger.warning(
                "Line %d: could not resolve item_code for '%s'",
                idx + 1, li.description,
            )
