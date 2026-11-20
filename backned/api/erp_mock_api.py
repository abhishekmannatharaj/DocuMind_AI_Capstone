"""
SAP ERP Mock Service (PO & Vendor Master)
=========================================
A lightweight **FastAPI** application that simulates an enterprise
resource planning (ERP) system for SAP Code Unnati 5.0 (DocuMind AI).
It serves read-only endpoints backed by JSON files in ``data/ERP_mockdata/``:

Endpoints
---------
GET  /api/erp/po/{po_number}        → Purchase Order with line items
GET  /api/erp/vendor/{vendor_id}    → Vendor master record (incl. GSTIN)
GET  /api/erp/sku/{item_code}       → SKU / item master record
GET  /api/erp/po                    → List all POs (for debugging)
GET  /api/erp/vendors               → List all vendors
GET  /api/erp/skus                  → List all SKUs

Usage
-----
    # Start the server (default port 8001)
    python -m backned.api.erp_mock_api

    # Or via uvicorn
    uvicorn backned.api.erp_mock_api:app --port 8001 --reload
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)

# 
# Path constants
# 
BASE_DIR = Path(__file__).resolve().parent.parent  # → backned/
ERP_DATA_DIR = BASE_DIR / "data" / "ERP_mockdata"

# 
# Load ERP data from JSON files (once at startup)
# 
def _load_json(filename: str) -> list[dict]:
    fp = ERP_DATA_DIR / filename
    if not fp.exists():
        logger.warning("ERP data file not found: %s", fp)
        return []
    with open(fp, "r", encoding="utf-8") as fh:
        return json.load(fh)


_PO_RECORDS: list[dict] = _load_json("PO Records.json")
_VENDORS: list[dict] = _load_json("vendors.json")
_SKU_MASTER: list[dict] = _load_json("sku_master.json")

# Build lookup indices for fast querying
_PO_INDEX: dict[str, dict] = {
    po["po_number"]: po for po in _PO_RECORDS
}
_VENDOR_INDEX: dict[str, dict] = {
    v["vendor_id"]: v for v in _VENDORS
}
_SKU_INDEX: dict[str, dict] = {
    s["item_code"]: s for s in _SKU_MASTER
}


# 
# FastAPI app
# 
app = FastAPI(
    title="SAP ERP Mock Service (PO & Vendor Master)",
    description=(
        "Simulated SAP ERP system API for DocuMind AI. "
        "Serves Purchase Orders, Vendor Master (with GSTIN), and Material/SKU Master data."
    ),
    version="1.0.0",
)


#  Health check 
@app.get("/api/erp/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "po_records": len(_PO_RECORDS),
        "vendors": len(_VENDORS),
        "skus": len(_SKU_MASTER),
    }


#  Purchase Orders 
@app.get("/api/erp/po")
def list_purchase_orders():
    """List all Purchase Orders."""
    return _PO_RECORDS


@app.get("/api/erp/po/{po_number}")
def get_purchase_order(po_number: str):
    """Get a specific Purchase Order by PO number.

    Returns the PO record with vendor_id and all line items
    (item_code, description, qty, unit_price, currency).
    """
    po = _PO_INDEX.get(po_number.upper()) or _PO_INDEX.get(po_number)
    if not po:
        raise HTTPException(
            status_code=404,
            detail=f"Purchase Order '{po_number}' not found in ERP system",
        )
    return po


#  Vendors 
@app.get("/api/erp/vendors")
def list_vendors():
    """List all vendors."""
    return _VENDORS


@app.get("/api/erp/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    """Get a vendor master record by vendor ID."""
    vendor = _VENDOR_INDEX.get(vendor_id.upper()) or _VENDOR_INDEX.get(vendor_id)
    if not vendor:
        raise HTTPException(
            status_code=404,
            detail=f"Vendor '{vendor_id}' not found in ERP system",
        )
    return vendor


#  SKU Master 
@app.get("/api/erp/skus")
def list_skus():
    """List all SKU / item master records."""
    return _SKU_MASTER


@app.get("/api/erp/sku/{item_code}")
def get_sku(item_code: str):
    """Get a SKU / item master record by item code."""
    sku = _SKU_INDEX.get(item_code.upper()) or _SKU_INDEX.get(item_code)
    if not sku:
        raise HTTPException(
            status_code=404,
            detail=f"SKU '{item_code}' not found in ERP system",
        )
    return sku


# 
# CLI entry-point
# 
if __name__ == "__main__":
    import uvicorn

    print("Starting Mock ERP API on http://localhost:8001")
    print(f"PO Records : {len(_PO_RECORDS)}")
    print(f"Vendors : {len(_VENDORS)}")
    print(f"SKUs  : {len(_SKU_MASTER)}")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
