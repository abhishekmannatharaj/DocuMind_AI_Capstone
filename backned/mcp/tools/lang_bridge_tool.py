"""Lang-Bridge Tool — translates extracted invoice content into English using Azure OpenAI GPT-4o.

Passes through English invoices unchanged. Exposes a LangChain @tool for use by the Translation Agent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

from backned.models.extraction import ExtractedInvoice, ExtractionResult
from backned.models.translation import (
    TranslatedInvoice,
    TranslatedLineItem,
    TranslationResult,
)
from backned.config.settings import (
    AZURE_AI_ENDPOINT,
    API_KEY,
    API_VERSION,
    DEPLOYMENT_NAME,
)
from backned.services.llm_provider import (
    call_chat_llm,
    deterministic_translate,
)

logger = logging.getLogger(__name__)

#  Module-level singleton variable 
_client: AzureOpenAI | None = None


# 
# Lang-Bridge Tool  (LangChain @tool)
# 
@tool
def lang_bridge_tool(extraction_result: dict) -> dict:
    """Lang-Bridge Tool — Translates extracted invoice content from
    any language into English using Azure OpenAI GPT-4o.

    If the source language is already English, the data is passed
    through unchanged (no API call).  For non-English invoices, every
    free-text field is translated while numeric / code fields are
    preserved as-is.

    Args:
        extraction_result: A dict serialised from ExtractionResult
            (output of the Data Harvester Tool).

    Returns:
        dict serialised from TranslationResult.
    """
    #  Rehydrate the ExtractionResult 
    try:
        ext = ExtractionResult(**extraction_result)
    except Exception as exc:
        return TranslationResult(
            email_id=extraction_result.get("email_id", "unknown"),
            filename=extraction_result.get("filename", "unknown"),
            file_path=extraction_result.get("file_path", ""),
            translation_errors=[f"Invalid extraction result: {exc}"],
        ).model_dump()

    #  No invoice data → nothing to translate 
    if not ext.extracted_invoice:
        return TranslationResult(
            email_id=ext.email_id,
            filename=ext.filename,
            file_path=ext.file_path,
            source_language=None,
            was_translated=False,
            translation_errors=["No extracted invoice data to translate"],
        ).model_dump()

    inv = ext.extracted_invoice
    src_lang = (inv.detected_language or "en").lower().strip()

    #  Already English → pass-through 
    if src_lang == "en":
        logger.info(
            "%s — already English, skipping translation", ext.filename
        )
        translated = _passthrough_invoice(inv)
        return TranslationResult(
            email_id=ext.email_id,
            filename=ext.filename,
            file_path=ext.file_path,
            source_language=src_lang,
            was_translated=False,
            translated_invoice=translated,
        ).model_dump()

    #  Translate with LLM or deterministic fallback 
    logger.info(
        "%s — translating from '%s' to English …",
        ext.filename, src_lang,
    )

    errors: list[str] = []
    try:
        translated_data = _translate_with_gpt(inv, src_lang)
        translated_invoice = _build_translated_invoice(
            inv, translated_data, src_lang
        )
    except Exception as exc:
        logger.info("LLM translation not available (%s); using deterministic translation", exc)
        # Fall back to deterministic translation
        t_items = []
        for li in inv.line_items:
            tr_desc, conf = deterministic_translate(li.description or "", src_lang)
            t_items.append({
                "item_code": li.item_code,
                "description": tr_desc,
            })
        translated_data = {
            "invoice_date": inv.invoice_date,
            "vendor_name": inv.vendor_name,
            "line_items": t_items,
            "additional_fields": inv.additional_fields,
        }
        translated_invoice = _build_translated_invoice(inv, translated_data, src_lang)

    return TranslationResult(
        email_id=ext.email_id,
        filename=ext.filename,
        file_path=ext.file_path,
        source_language=src_lang,
        was_translated=len(errors) == 0,
        translated_invoice=translated_invoice,
        translation_errors=errors,
    ).model_dump()


# 
#  HELPER FUNCTIONS  (called by the tool above)
# 


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
# GPT-4o translation prompt & call
# 

_TRANSLATION_SYSTEM_PROMPT = """\
You are an expert multilingual translator specialising in business and
financial documents.  Your task is to translate invoice content from the
source language into **English**.

IMPORTANT RULES:
1. Translate ALL free-text / natural-language fields INCLUDING dates.
2. Convert dates to English format (e.g. "10 de abril de 2025" → "April 10, 2025",
   "25. Mai 2025" → "May 25, 2025"). Always output dates as "%B %d, %Y" format
   (e.g. "January 15, 2025").
3. Do NOT translate or modify:
   - Invoice numbers, PO numbers, vendor IDs, SKU/item codes
   - Numeric values (amounts, quantities, prices)
   - Currency codes
4. Preserve the EXACT meaning of every term — do not paraphrase.
5. Return ONLY valid JSON — no markdown fences, no commentary.

Return JSON in this schema:
{
  "invoice_date": "<date translated to English format, e.g. 'April 10, 2025'>",
  "vendor_name": "<translated vendor name, or original if already English>",
  "line_items": [
    {
      "item_code": "<unchanged>",
      "description": "<translated description>"
    }
  ],
  "additional_fields": {
    "<key>": "<translated value>"
  }
}

Rules for the response:
- line_items must have the SAME number of entries and SAME order as the
  input — match them by position.
- If a value is already in English, return it unchanged.
- Do NOT add extra keys.
"""


def _translate_with_gpt(
    invoice: ExtractedInvoice, source_language: str
) -> dict:
    """Send the translatable fields to GPT-4o and return the
    translated JSON dict."""
    client = _get_client()

    # Build a compact representation of what needs translating
    translatable = {
        "source_language": source_language,
        "invoice_date": invoice.invoice_date,
        "vendor_name": invoice.vendor_name,
        "line_items": [
            {
                "item_code": li.item_code,
                "description": li.description,
            }
            for li in invoice.line_items
        ],
        "additional_fields": invoice.additional_fields or {},
    }

    user_msg = (
        f"Translate the following invoice content from "
        f"'{source_language}' to English.\n\n"
        f"{json.dumps(translatable, ensure_ascii=False, indent=2)}"
    )

    messages = [
        {"role": "system", "content": _TRANSLATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    llm_resp = call_chat_llm(messages, max_tokens=4096, temperature=0.0)
    if not llm_resp:
        raise RuntimeError("No LLM response available")

    content = llm_resp.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
    if content.endswith("```"):
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    return json.loads(content)


# 
# Build TranslatedInvoice from GPT response + original
# 

def _build_translated_invoice(
    original: ExtractedInvoice,
    translated_data: dict,
    source_language: str,
) -> TranslatedInvoice:
    """Merge the original numeric/code fields with the GPT-translated
    text fields into a TranslatedInvoice model."""

    #  Translated line items 
    translated_items_raw = translated_data.get("line_items", [])
    line_items: list[TranslatedLineItem] = []

    for i, orig_li in enumerate(original.line_items):
        # Match by position
        trans_li = (
            translated_items_raw[i]
            if i < len(translated_items_raw)
            else {}
        )
        line_items.append(TranslatedLineItem(
            item_code=orig_li.item_code,
            original_description=orig_li.description,
            translated_description=trans_li.get(
                "description", orig_li.description
            ),
            qty=orig_li.qty,
            unit_price=orig_li.unit_price,
            total=orig_li.total,
        ))

    return TranslatedInvoice(
        # Codes & numbers — unchanged
        invoice_no=original.invoice_no,
        invoice_date=translated_data.get(
            "invoice_date", original.invoice_date
        ),
        po_number=original.po_number,
        vendor_id=original.vendor_id,
        currency=original.currency,
        total_amount=original.total_amount,
        tax_amount=original.tax_amount,
        subtotal=original.subtotal,
        # Translated text fields
        original_vendor_name=original.vendor_name,
        translated_vendor_name=translated_data.get(
            "vendor_name", original.vendor_name
        ),
        line_items=line_items,
        # Language info
        source_language=source_language,
        target_language="en",
        was_translated=True,
        # Additional fields
        original_additional_fields=original.additional_fields or {},
        translated_additional_fields=translated_data.get(
            "additional_fields", original.additional_fields or {}
        ),
    )


# 
# Pass-through: already English (no API call)
# 

def _passthrough_invoice(invoice: ExtractedInvoice) -> TranslatedInvoice:
    """Wrap an already-English invoice into TranslatedInvoice without
    calling the LLM.  original == translated for every field."""
    line_items = [
        TranslatedLineItem(
            item_code=li.item_code,
            original_description=li.description,
            translated_description=li.description,  # same
            qty=li.qty,
            unit_price=li.unit_price,
            total=li.total,
        )
        for li in invoice.line_items
    ]

    return TranslatedInvoice(
        invoice_no=invoice.invoice_no,
        invoice_date=invoice.invoice_date,
        po_number=invoice.po_number,
        vendor_id=invoice.vendor_id,
        original_vendor_name=invoice.vendor_name,
        translated_vendor_name=invoice.vendor_name,  # same
        currency=invoice.currency,
        total_amount=invoice.total_amount,
        tax_amount=invoice.tax_amount,
        subtotal=invoice.subtotal,
        line_items=line_items,
        source_language=invoice.detected_language or "en",
        target_language="en",
        was_translated=False,
        original_additional_fields=invoice.additional_fields or {},
        translated_additional_fields=invoice.additional_fields or {},  # same
    )
