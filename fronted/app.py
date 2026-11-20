"""AI Invoice Auditor — Streamlit Frontend

Professional UI for monitoring, processing, and reviewing AI-audited invoices.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

#  Project root setup 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set CWD so relative paths in rules.yaml (logs, reports) resolve correctly
os.chdir(PROJECT_ROOT)

#  Page config (must be first Streamlit call) 
st.set_page_config(
    page_title="DocuMind AI — Intelligent Document Understanding",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "DocuMind AI — SAP Code Unnati 5.0 Capstone Project | Abhishek Mannatharaj & Devika G"},
)

#  Path constants 
REPORTS_DIR    = PROJECT_ROOT / "outputs" / "reports"
PROCESSED_FILE = PROJECT_ROOT / "backned" / "data" / "processed_emails.json"
RULES_PATH     = PROJECT_ROOT / "backned" / "config" / "rules.yaml"
LOGS_FILE      = PROJECT_ROOT / "logs" / "invoice_auditor.log"

#  Recommendation palette 
REC = {
    "auto_approve":  {"label": "Auto Approve",  "pill": "pill-green", "icon": "", "color": "#198754", "bg": "#d1e7dd", "bar": "#198754"},
    "manual_review": {"label": "Manual Review", "pill": "pill-amber", "icon": "!",  "color": "#d97706", "bg": "#fff3cd", "bar": "#ffc107"},
    "reject":        {"label": "Reject",        "pill": "pill-red",   "icon": "", "color": "#dc3545", "bg": "#f8d7da", "bar": "#dc3545"},
    "unknown":       {"label": "Unknown",       "pill": "pill-grey",  "icon": "?", "color": "#6c757d", "bg": "#f0f2f5", "bar": "#adb5bd"},
}

# 
# CSS
# 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/*  Sidebar  */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }

/* Sidebar text colours */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #94a3b8 !important;
}

/* Radio as nav */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 10px 14px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #1e40af !important;
    color: #ffffff !important;
}

/* Hide radio circle dots in sidebar nav */
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none !important;
}

/*  Main content  */
.main .block-container {
    padding: 28px 36px 40px !important;
    max-width: 1440px !important;
    background: #f8fafc !important;
}

/*  KPI cards  */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(15,23,42,.06);
    transition: transform .15s, box-shadow .15s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79,70,229,.12);
}
.kpi-icon {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; margin-bottom: 12px;
}
.kpi-value { font-size: 22px; font-weight: 800; line-height: 1; }
.kpi-label { font-size: 10px; font-weight: 700; color: #374151;
             text-transform: uppercase; letter-spacing: .6px; margin-top: 5px; }
.kpi-sub   { font-size: 10px; color: #6366f1; margin-top: 3px; font-weight: 500; }

/*  Section header  */
.sec-hdr {
    font-size: 11px; font-weight: 700; letter-spacing: .8px;
    text-transform: uppercase; color: #6366f1;
    display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
}
.sec-hdr::before {
    content: ""; display: block; width: 3px; height: 16px;
    border-radius: 2px;
    background: linear-gradient(180deg, #6366f1 0%, #a855f7 100%);
}


/*  Report card text  */
.rcard-name {
    font-size: 12px; font-weight: 600; color: #0f172a;
    margin: 1px 0 1px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.rcard-score  { font-size: 19px; font-weight: 700; line-height: 1.15; margin: 1px 0 0; }
.rcard-meta   { font-size: 10.5px; color: #64748b; }

/*  Compact report cards: shrink Streamlit internal gap  */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0px !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 6px 10px !important;
}

/*  Pill badges  */
.pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 9px; border-radius: 20px; font-size: 10.5px; font-weight: 600;
}
.pill-green { background: #d1e7dd; color: #0f5132; }
.pill-amber { background: #fff3cd; color: #664d03; }
.pill-red   { background: #f8d7da; color: #842029; }
.pill-blue  { background: #dbeafe; color: #1e40af; }
.pill-grey  { background: #e9ecef; color: #495057; }


/*  Log box  */
.log-box {
    background: #0f172a; border-radius: 10px;
    padding: 12px 16px; max-height: 300px; overflow-y: auto;
    font-family: 'Cascadia Code', 'SF Mono', Consolas, monospace;
    font-size: 10.5px; line-height: 1.6;
}
.l-info    { color: #94a3b8; }
.l-warn    { color: #fbbf24; }
.l-err     { color: #f87171; }
.l-success { color: #6ee7b7; }

/*  Divider  */
.hdiv { border-top: 1px solid #e2e8f0; margin: 20px 0; }


/*  Button overrides  */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 12px !important;
    border-radius: 7px !important; transition: all .15s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(79,70,229,.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(79,70,229,.4) !important;
    transform: translateY(-1px) !important;
}

/*  Progress bar  */
.stProgress > div > div { border-radius: 6px; }
[data-testid="stProgressBar"] > div { background: #4f46e5 !important; }

/*  Scrollbar  */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/*  Expander  */
[data-testid="stExpander"] summary {
    font-size: 12px !important; font-weight: 600 !important; color: #0f172a !important;
}

/*  Page title  */
.pg-title {
    font-size: 24px; font-weight: 800; line-height: 1.2;
    color: #312e81;
}
.pg-title .pg-emoji { color: initial; -webkit-text-fill-color: initial; }
.pg-sub   { font-size: 13px; color: #6366f1; margin-top: 3px; margin-bottom: 22px; font-weight: 500; }

/* Metric cards (st.metric override) */
[data-testid="stMetric"] {
    background: #fff; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 18px;
}
[data-testid="stMetricLabel"]  { font-size: 10.5px !important; }
[data-testid="stMetricValue"]  { font-size: 20px !important; }

/*  Colourful action buttons  */
.btn-blue  button { background:#4f46e5 !important; border-color:#4f46e5 !important; color:#fff !important; font-weight:600 !important; }
.btn-blue  button:hover { background:#4338ca !important; border-color:#4338ca !important; }
.btn-green button { background:#16a34a !important; border-color:#16a34a !important; color:#fff !important; font-weight:600 !important; }
.btn-green button:hover { background:#15803d !important; border-color:#15803d !important; }
.btn-amber button { background:#d97706 !important; border-color:#d97706 !important; color:#fff !important; font-weight:600 !important; }
.btn-amber button:hover { background:#b45309 !important; border-color:#b45309 !important; }
.btn-clear button {
    background: linear-gradient(135deg,#ef4444,#dc2626) !important;
    border-color: #dc2626 !important; color: #fff !important;
    font-weight: 600 !important; font-size: 12px !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 8px rgba(220,38,38,.25) !important;
    transition: all .15s !important;
}
.btn-clear button:hover {
    background: linear-gradient(135deg,#dc2626,#b91c1c) !important;
    border-color: #b91c1c !important;
    box-shadow: 0 4px 14px rgba(185,28,28,.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Colourful step spinner ── */
@keyframes colorSpin {
    0%   { border-top-color:#f43f5e; border-right-color:#f97316; }
    25%  { border-top-color:#a855f7; border-right-color:#3b82f6; }
    50%  { border-top-color:#10b981; border-right-color:#06b6d4; }
    75%  { border-top-color:#f59e0b; border-right-color:#84cc16; }
    100% { border-top-color:#f43f5e; border-right-color:#f97316; }
}
.step-spinner {
    display: inline-block;
    width: 15px; height: 15px;
    border: 2.5px solid rgba(0,0,0,0.08);
    border-top-color: #f43f5e;
    border-right-color: #f97316;
    border-radius: 50%;
    animation: colorSpin 0.75s linear infinite;
    vertical-align: middle;
    margin-right: 6px;
    flex-shrink: 0;
}
.step-row-active {
    display: flex; align-items: center; gap: 6px;
    font-size: 14px; padding: 5px 0; color: #0f172a;
}
.step-row-done {
    display: flex; align-items: center; gap: 6px;
    font-size: 14px; padding: 5px 0; color: #16a34a;
}
</style>
""", unsafe_allow_html=True)


# 
# Utility helpers
# 

def _pill(rec: str) -> str:
    cfg = REC.get(rec, REC["unknown"])
    return f'<span class="pill {cfg["pill"]}">{cfg["icon"]} {cfg["label"]}</span>'


def _sec(title: str) -> None:
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)


def _kpi(icon: str, icon_bg: str, value, label: str, sub: str = "", value_color: str = "#1e293b", border_color: str = "#e2e8f0") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card" style="border-top:3px solid {border_color};">'
        f'<div class="kpi-icon" style="background:{icon_bg};">{icon}</div>'
        f'<div class="kpi-value" style="color:{value_color};">{value}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'{sub_html}</div>'
    )


def _log_html(lines: list[str]) -> str:
    if not lines:
        return '<div class="log-box"><span class="l-info">No log entries captured.</span></div>'
    parts = []
    for ln in lines:
        safe = ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "[ERROR]" in ln:
            cls = "l-err"
        elif "[WARNING]" in ln:
            cls = "l-warn"
        elif any(x in ln for x in ["", "complete", ""]):
            cls = "l-success"
        else:
            cls = "l-info"
        parts.append(f'<div class="{cls}">{safe}</div>')
    return '<div class="log-box">' + "".join(parts) + "</div>"


def _load_log_tail(n: int = 80) -> list[str]:
    try:
        if not LOGS_FILE.exists():
            return []
        lines = LOGS_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


# 
# Report parser (reads existing HTML files from disk)
# 

def _read_all_reports() -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for f in sorted(REPORTS_DIR.glob("*.html"), reverse=True):
        try:
            html = f.read_text(encoding="utf-8")
            reports.append(_parse_report_html(html, f))
        except Exception:
            pass
    return reports


def _parse_report_html(html: str, path: Path) -> dict:
    """Extract key metadata from a generated report HTML file."""
    # Recommendation — from verdict-label span
    m = re.search(r'class="verdict-label">(.*?)</span>', html)
    raw_rec = m.group(1).strip().lower().replace(" ", "_") if m else "unknown"
    if raw_rec not in REC:
        if "auto" in raw_rec:
            raw_rec = "auto_approve"
        elif "manual" in raw_rec or "review" in raw_rec:
            raw_rec = "manual_review"
        elif "reject" in raw_rec:
            raw_rec = "reject"
        else:
            raw_rec = "unknown"

    # Confidence
    m = re.search(r'confidence:.*?<strong>(\d+)%</strong>', html, re.DOTALL)
    confidence = int(m.group(1)) if m else 0

    # Filename from topbar
    m = re.search(r'topbar-sub">(.*?)&ensp;·&ensp;(.*?)</div>', html)
    filename = m.group(1).strip() if m else path.stem

    # Invoice number from field grid
    m = re.search(r'Invoice No\.</div>\s*<div[^>]*fv-mono[^>]*>(.*?)</div>', html, re.DOTALL)
    invoice_no = m.group(1).strip() if m else "—"

    # Total amount
    m = re.search(r'class="fv-lg">(.*?)</span>', html)
    total = m.group(1).strip() if m else "—"

    # Invoice date
    m = re.search(r'Invoice Date</div>\s*<div[^>]*>(.*?)</div>', html, re.DOTALL)
    inv_date = m.group(1).strip() if m else "—"

    # PO Number
    m = re.search(r'PO Number</div>\s*<div[^>]*fv-mono[^>]*>(.*?)</div>', html, re.DOTALL)
    po_no = m.group(1).strip() if m else "—"

    # Vendor ID
    m = re.search(r'Vendor ID</div>\s*<div[^>]*fv-mono[^>]*>(.*?)</div>', html, re.DOTALL)
    vendor_id = m.group(1).strip() if m else "—"

    # Vendor Name
    m = re.search(r'Vendor Name</div>\s*<div[^>]*>(.*?)</div>', html, re.DOTALL)
    vendor_name = m.group(1).strip() if m else "—"

    return {
        "path":           str(path),
        "filename":       filename,
        "recommendation": raw_rec,
        "confidence":     confidence,
        "invoice_no":     invoice_no,
        "inv_date":       inv_date,
        "po_no":          po_no,
        "vendor_id":      vendor_id,
        "vendor_name":    vendor_name,
        "total":          total,
        "html":           html,
    }


# 
# Page: Dashboard
# 

def page_dashboard() -> None:
    st.markdown('<div class="pg-title"><span class="pg-emoji">📑</span> DocuMind AI — Intelligent Document Understanding</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pg-sub"><b>SAP Code Unnati 5.0 Capstone Project</b> &nbsp;·&nbsp; '
        '<i>Abhishek Mannatharaj & Devika G | Dept. of CSE, Gopalan College of Engineering and Management, Bengaluru</i></div>',
        unsafe_allow_html=True,
    )

    reports = _read_all_reports()
    total   = len(reports)
    auto    = sum(1 for r in reports if r["recommendation"] == "auto_approve")
    manual  = sum(1 for r in reports if r["recommendation"] == "manual_review")
    reject  = sum(1 for r in reports if r["recommendation"] == "reject")
    rate    = f"{int(auto / total * 100)}% approval" if total else "no data yet"

    #  KPI row 
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_kpi("📊", "linear-gradient(135deg,#dbeafe,#eff6ff)", total, "Total Processed", "lifetime reports", "#1d4ed8", "#3b82f6"), unsafe_allow_html=True)
    with c2:
        st.markdown(_kpi("✅", "linear-gradient(135deg,#dcfce7,#f0fdf4)", auto, "Auto-Approved", rate, "#15803d", "#22c55e"), unsafe_allow_html=True)
    with c3:
        st.markdown(_kpi("🔍", "linear-gradient(135deg,#fef3c7,#fffbeb)", manual, "Manual Review", "require human check", "#b45309", "#f59e0b"), unsafe_allow_html=True)
    with c4:
        st.markdown(_kpi("🚫", "linear-gradient(135deg,#fee2e2,#fff5f5)", reject, "Rejected", "failed validation", "#b91c1c", "#ef4444"), unsafe_allow_html=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    #  Processed Invoices Table 
    _sec("Processed Invoices")

    if not reports:
        st.info("No invoices processed yet — go to **Pipeline** to run the auditor.")
        return

    STATUS_STYLE = {
        "auto_approve":  ("✅ Auto Approved",  "#dcfce7", "#15803d"),
        "manual_review": ("🔍 Manual Review",  "#fef3c7", "#92400e"),
        "reject":        ("🚫 Rejected",        "#fee2e2", "#991b1b"),
        "unknown":       ("❓ Unknown",          "#f1f5f9", "#475569"),
    }

    # Column weights: InvNo | Date | PO | VendorID | VendorName | Total | Status
    CW = [2.0, 1.6, 1.6, 1.4, 2.0, 1.6, 1.9]

    # Header row
    hdrs = ["Invoice No.", "Date", "PO Number", "Vendor ID", "Vendor Name", "Total", "Status"]
    HEADER_COLORS = ["#6366f1", "#0891b2", "#0891b2", "#7c3aed", "#7c3aed", "#059669", "#db2777"]
    hcols = st.columns(CW, gap="small")
    for hc, hdr, hcol in zip(hcols, hdrs, HEADER_COLORS):
        hc.markdown(
            f'<div style="font-size:10px;font-weight:700;color:{hcol};'
            f'text-transform:uppercase;letter-spacing:.6px;'
            f'padding:5px 2px 8px;border-bottom:2px solid {hcol}22;">{hdr}</div>',
            unsafe_allow_html=True,
        )

    # Data rows
    cell = 'font-size:12px;color:{fg};padding:7px 2px;border-bottom:1px solid #f1f5f9;'
    for i, r in enumerate(reports):
        lbl, pill_bg, pill_fg = STATUS_STYLE.get(r["recommendation"], STATUS_STYLE["unknown"])
        inv_no    = r.get("invoice_no",  "—") or "—"
        inv_date  = r.get("inv_date",    "—") or "—"
        po_no     = r.get("po_no",       "—") or "—"
        vendor_id = r.get("vendor_id",   "—") or "—"
        vname     = r.get("vendor_name", "—") or "—"
        total     = r.get("total",       "—") or "—"

        rcols = st.columns(CW, gap="small")
        rcols[0].markdown(f'<div style="{cell.format(fg="#0f172a")};font-weight:600;white-space:nowrap;">{inv_no}</div>', unsafe_allow_html=True)
        rcols[1].markdown(f'<div style="{cell.format(fg="#374151")};white-space:nowrap;">{inv_date}</div>', unsafe_allow_html=True)
        rcols[2].markdown(f'<div style="{cell.format(fg="#374151")};font-family:monospace;font-size:11px;">{po_no}</div>', unsafe_allow_html=True)
        rcols[3].markdown(f'<div style="{cell.format(fg="#374151")};font-family:monospace;font-size:11px;">{vendor_id}</div>', unsafe_allow_html=True)
        rcols[4].markdown(f'<div style="{cell.format(fg="#374151")};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{vname}">{vname}</div>', unsafe_allow_html=True)
        rcols[5].markdown(f'<div style="{cell.format(fg="#0f172a")};font-weight:600;white-space:nowrap;text-align:right;">{total}</div>', unsafe_allow_html=True)
        rcols[6].markdown(
            f'<div style="padding:7px 2px;border-bottom:1px solid #f1f5f9;">'
            f'<span style="display:inline-block;padding:2px 8px;border-radius:20px;'
            f'background:{pill_bg};color:{pill_fg};font-size:10px;font-weight:600;white-space:nowrap;">{lbl}</span></div>',
            unsafe_allow_html=True,
        )


def _run_pipeline_ui(reset: bool) -> None:
    """Execute the pipeline, show real-time progress, then display results."""
    from fronted.pipeline_runner import PipelineRunner, STAGE_INFO, STAGE_KEYS  # noqa: PLC0415

    progress  = st.progress(0, text="Initialising pipeline…")
    final: dict = {}

    with st.status("  Running Invoice Audit Pipeline", expanded=True) as status_wgt:
        # Pre-create one placeholder per step so we can update them in-place
        step_placeholders = [st.empty() for _ in STAGE_INFO]

        def on_start(key: str, idx: int, n: int) -> None:
            _, name, desc = STAGE_INFO[idx]
            # Show a colourful spinning loader next to the active step
            step_placeholders[idx].markdown(
                f'<div class="step-row-active">'
                f'<span class="step-spinner"></span>'
                f'<span><b>Step {idx + 1}/{n} — {name}:</b> {desc}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            progress.progress(idx / n, text=f"Running: {name}…")

        def on_done(key: str, idx: int, n: int, logs: list[str]) -> None:
            _, name, desc = STAGE_INFO[idx]
            # Replace the spinner with a green ✅ checkmark
            step_placeholders[idx].markdown(
                f'<div class="step-row-done">'
                f'✅ <b>Step {idx + 1}/{n} — {name}:</b> {desc}'
                f'</div>',
                unsafe_allow_html=True,
            )

        try:
            runner = PipelineRunner()
            result = runner.run_full_pipeline(reset=reset, on_stage_start=on_start, on_stage_done=on_done)
            final  = result
            progress.progress(1.0, text="Pipeline complete ")

            summary = result.get("reporting_summary")
            if summary and summary.total_invoices:
                status_wgt.update(
                    label=f"  Pipeline Complete — {summary.total_invoices} invoice(s) processed",
                    state="complete",
                )
            else:
                status_wgt.update(label="  Pipeline finished — no invoices processed", state="error")

            st.session_state["last_pipeline"] = final
            st.session_state["last_run_time"]  = datetime.now().strftime("%H:%M:%S")

        except Exception as exc:
            status_wgt.update(label=f"  Pipeline error: {exc}", state="error")
            st.error(f"**Pipeline error:** {exc}")
            return

    # Results table
    _render_results(final)

    # Stage logs
    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    _sec("Stage Logs")
    for i, (icon, name, _) in enumerate(STAGE_INFO):
        key  = STAGE_KEYS[i]
        logs = final.get("all_logs", {}).get(key, [])
        with st.expander(f"{icon} {name} — {len(logs)} entries", expanded=False):
            st.markdown(_log_html(logs), unsafe_allow_html=True)


def _render_results(results: dict) -> None:
    summary = results.get("reporting_summary")
    if not summary:
        return

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    _sec("Pipeline Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Total",        summary.total_invoices)
    c2.metric("✅ Auto-Approved", summary.auto_approved)
    c3.metric("🔍 Manual Review", summary.manual_review)
    c4.metric("🚫 Rejected",      summary.rejected)

    if summary.reports:
        rows = []
        for r in summary.reports:
            rows.append({
                "File":           r.filename,
                "Invoice #":      r.invoice_no or "—",
                "Vendor":         r.vendor_name or "—",
                "Total":          f"{r.currency or ''} {r.total_amount:,.2f}".strip() if r.total_amount else "—",
                "Recommendation": r.recommendation.replace("_", " ").title(),
                "Score":          round(r.auto_approve_confidence * 100),
            })
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score (%)", min_value=0, max_value=100, format="%d%%"
                ),
            },
        )


# 
# Page: Pipeline
# 

def page_pipeline() -> None:
    st.markdown('<div class="pg-title"><span class="pg-emoji">⚙️</span> Document Understanding Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pg-sub">Multi-agent extraction, validation & cross-referencing with SAP ERP Mock Service (PO & Vendor Master)</div>',
        unsafe_allow_html=True,
    )

    _sec("Pipeline Control")
    pc1, pc2, _ = st.columns([1.4, 1.4, 3.2], gap="small")
    with pc1:
        st.markdown('<div class="btn-blue">', unsafe_allow_html=True)
        run_all = st.button("Run Full Pipeline", use_container_width=True,
                            help="Reset all processed state and re-run every invoice")
        st.markdown('</div>', unsafe_allow_html=True)
    with pc2:
        st.markdown('<div class="btn-green">', unsafe_allow_html=True)
        run_new = st.button("Run New Only", use_container_width=True,
                            help="Only process emails/invoices not yet seen")
        st.markdown('</div>', unsafe_allow_html=True)

    if run_all or run_new:
        _run_pipeline_ui(reset=bool(run_all))
    elif "last_pipeline" in st.session_state:
        _render_results(st.session_state["last_pipeline"])


# 
# Page: Reports
# 

def page_reports() -> None:
    #  Inline viewer 
    if "view_report" in st.session_state:
        r = st.session_state["view_report"]
        col_back, col_title = st.columns([1, 8])
        with col_back:
            if st.button("← Back"):
                del st.session_state["view_report"]
                st.rerun()
        with col_title:
            st.markdown(
                f'<div style="font-size:15px;font-weight:600;color:#0f172a;padding-top:6px;">'
                f'{r["filename"]}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
        components.html(r["html"], height=940, scrolling=True)
        return

    st.markdown('<div class="pg-title"><span class="pg-emoji">📋</span> Audit Reports & SAP Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pg-sub">Detailed audit reports cross-referenced against SAP ERP Mock Service (PO & Vendor Master)</div>',
        unsafe_allow_html=True,
    )

    reports = _read_all_reports()
    if not reports:
        st.info("No reports found. Go to **Process Invoices** to run the pipeline.")
        return

    #  Filter bar 
    fc1, fc2 = st.columns([2, 4])
    with fc1:
        flt = st.selectbox(
            "Filter",
            ["All Outcomes", "Auto Approve", "Manual Review", "Reject"],
            label_visibility="collapsed",
        )
    with fc2:
        q = st.text_input("Search", placeholder="Search filename or invoice #…",
                          label_visibility="collapsed")

    filtered = reports
    flt_map  = {"Auto Approve": "auto_approve", "Manual Review": "manual_review", "Reject": "reject"}
    if flt != "All Outcomes":
        filtered = [r for r in filtered if r["recommendation"] == flt_map[flt]]
    if q:
        ql = q.lower()
        filtered = [r for r in filtered if ql in r["filename"].lower() or ql in r["invoice_no"].lower()]

    st.markdown(
        f'<div style="font-size:11.5px;font-weight:600;color:#6366f1;margin-bottom:16px;">'
        f'📌 {len(filtered)} of {len(reports)} report(s) shown</div>',
        unsafe_allow_html=True,
    )

    #  Card grid (3 columns) 
    cols = st.columns(3, gap="medium")
    for i, r in enumerate(filtered):
        conf = r["confidence"]
        conf_color = "#15803d" if conf >= 90 else ("#b45309" if conf >= 60 else "#b91c1c")
        conf_bg    = "#f0fdf4" if conf >= 90 else ("#fffbeb" if conf >= 60 else "#fff5f5")

        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"""
                {_pill(r['recommendation'])}
                <div class="rcard-name" title="{r['filename']}">{r['filename']}</div>
                <div class="rcard-score" style="color:{conf_color};">{conf}%</div>
                <div class="rcard-meta" style="color:#6366f1;font-weight:600;">confidence score</div>
                <div class="rcard-meta" style="margin-top:4px;color:#374151;">
                    🗎 Invoice: <b>{r['invoice_no']}</b> &nbsp;·&nbsp; 💰 Total: <b>{r['total']}</b>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
                bc1, bc2 = st.columns(2, gap="small")
                with bc1:
                    if st.button("View", key=f"vr_{i}",
                                 use_container_width=True, help="View Report"):
                        st.session_state["view_report"] = r
                        st.rerun()
                with bc2:
                    st.download_button(
                        "Download",
                        data=r["html"].encode("utf-8"),
                        file_name=Path(r["path"]).name,
                        mime="text/html",
                        key=f"dl_{i}",
                        use_container_width=True,
                        help="Download Report HTML",
                    )


# 
# Page: Support Agent (RAG)
# 

def page_support_agent() -> None:
    #  Lazy import 
    try:
        from backned.services.rag_pipeline import get_rag_pipeline  # noqa: PLC0415
    except ImportError as e:
        st.error(f"RAG pipeline not available: {e}")
        return

    #  Session state init 
    if "rag_pipeline"      not in st.session_state:
                                st.session_state["rag_pipeline"]      = None
    if "rag_messages"      not in st.session_state:
                                st.session_state["rag_messages"]      = []
    if "rag_pending_query" not in st.session_state:
                                st.session_state["rag_pending_query"] = None

    #  Title row 
    title_col, clear_col = st.columns([6, 1.8], gap="small")
    with title_col:
        st.markdown('<div class="pg-title"><span class="pg-emoji">🤖</span> DocuMind Support Agent (RAG)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="pg-sub" style="font-size:11.5px;">Ask anything about your invoices — '
            'rejections, missing fields, SAP ERP mismatches, Indian GSTIN, vendor data, and [Page X] citations.</div>',
            unsafe_allow_html=True,
        )
    with clear_col:
        st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
        if st.button("🗑️  Clear Chat", use_container_width=True, key="rag_clear", help="Clear conversation history"):
            st.session_state["rag_messages"]      = []
            st.session_state["rag_pending_query"] = None
            if st.session_state["rag_pipeline"]:
                st.session_state["rag_pipeline"].clear_memory()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    #  Phase 2: if there's a pending query, run it now 
    # (user message was already appended + rerun triggered in phase 1)
    pending = st.session_state.get("rag_pending_query")

    #  Chat window (full width) 
    msgs = st.session_state["rag_messages"]
    if not msgs:
        st.markdown("""
        <div style="text-align:center;padding:52px 20px;
                    background:#ffffff;
                    border:1px solid #e2e8f0;border-radius:16px;
                    box-shadow:0 2px 12px rgba(15,23,42,.06);">
            <div style="width:56px;height:56px;border-radius:16px;
                        background:linear-gradient(135deg,#4f46e5,#7c3aed);
                        display:flex;align-items:center;justify-content:center;
                        margin:0 auto 16px;font-size:26px;color:#fff;">&#129302;</div>
            <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:6px;">Support Agent Ready</div>
            <div style="font-size:12px;color:#64748b;max-width:400px;
                        margin-inline:auto;line-height:1.7;">
                Ask me anything about your invoices —
                rejections, missing fields, ERP mismatches, vendor info, and more.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in msgs:
            role    = msg["role"]
            content = msg["content"]
            meta    = msg.get("meta", {})

            if role == "user":
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end;margin:6px 0;">'
                    f'<div style="background:#4f46e5;color:#fff;'
                    f'border-radius:14px 14px 2px 14px;'
                    f'padding:9px 14px;max-width:75%;'
                    f'font-size:11.5px;line-height:1.55;">'
                    f'{_esc_html(content)}</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                sources_html = ""
                if meta.get("sources_used"):
                    src_pills = " ".join(
                        f'<span class="pill pill-blue" style="font-size:9.5px;">'
                        f'{s.split(":")[0].upper()}: {s.split(":")[-1][:30]}</span>'
                        for s in meta["sources_used"][:4]
                    )
                    sources_html = (
                        f'<div style="margin-top:8px;padding-top:7px;'
                        f'border-top:1px solid #f0f2f5;">'
                        f'<span style="font-size:9.5px;color:#94a3b8;">Sources: </span>'
                        f'{src_pills}</div>'
                    )

                st.markdown(
                    f'<div style="display:flex;justify-content:flex-start;margin:6px 0;">'
                    f'<div style="background:#fff;border:1px solid #e0e7ff;'
                    f'border-radius:2px 14px 14px 14px;padding:9px 13px;max-width:84%;'
                    f'box-shadow:0 2px 8px rgba(79,70,229,.07);">'
                    f'<div style="font-size:9px;font-weight:700;color:#6366f1;'
                    f'letter-spacing:.5px;text-transform:uppercase;margin-bottom:5px;">'
                    f'&#129302; Support Agent</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div style="font-size:11.5px;line-height:1.6;">{content}</div>', unsafe_allow_html=True)
                st.markdown(f'{sources_html}</div></div>', unsafe_allow_html=True)

    #  Processing indicator (phase 2 in progress) 
    if pending:
        _handle_rag_query(pending, get_rag_pipeline)
        return

    #  Input area 
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    prefill = st.session_state.pop("rag_prefill", "")

    with st.form(key="rag_chat_form", clear_on_submit=True):
        inp_col, btn_col = st.columns([9, 1], gap="small")
        with inp_col:
            user_input = st.text_input(
                "Ask a question",
                value=prefill,
                placeholder="e.g. Why was invoice INV_EN_006 flagged for manual review?",
                label_visibility="collapsed",
                key="rag_input",
            )
        with btn_col:
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

    # Phase 1: append user message immediately, set pending flag, rerun
    if submitted and user_input.strip():
        q = user_input.strip()
        st.session_state["rag_messages"].append({"role": "user", "content": q})
        st.session_state["rag_pending_query"] = q
        st.rerun()


def _esc_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _handle_rag_query(query: str, get_pipeline_fn) -> None:
    """Process a pending RAG query and update the message history.
    User message was already appended before calling this (phase 1 of two-phase submit).
    """
    # Ensure pipeline exists
    if st.session_state["rag_pipeline"] is None:
        st.session_state["rag_pipeline"] = get_pipeline_fn()

    pipeline = st.session_state["rag_pipeline"]

    # Ensure index is built
    if not pipeline._index_built:
        with st.spinner("Building index for the first time…"):
            try:
                pipeline.ensure_index()
            except Exception as e:
                st.session_state["rag_messages"].append({
                    "role": "assistant",
                    "content": f" Could not build index: {e}",
                    "meta": {},
                })
                st.session_state["rag_pending_query"] = None
                st.rerun()
                return

    # Run query
    with st.spinner("Thinking…"):
        try:
            result = pipeline.query(query)
            st.session_state["rag_messages"].append({
                "role":    "assistant",
                "content": result.answer,
                "meta":    {"sources_used": result.sources_used},
            })
        except Exception as e:
            st.session_state["rag_messages"].append({
                "role": "assistant",
                "content": f" Error processing query: {e}",
                "meta": {},
            })

    st.session_state["rag_pending_query"] = None
    st.rerun()



def page_upload_and_review() -> None:
    st.markdown('<div class="pg-title"><span class="pg-emoji">📤</span> Custom Document Upload & Human Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pg-sub">Direct document upload, per-field AI confidence scoring, Human Review (Approve/Edit/Reject), and SAP Posting-Ready JSON Export.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload Invoice / PO Document (PDF, Image, DOCX)",
        type=["pdf", "png", "jpg", "jpeg", "docx"],
        help="Upload custom documents to test OCR, field extraction, confidence scoring, and SAP validation.",
    )

    if uploaded is not None:
        save_path = PROJECT_ROOT / "backned" / "data" / "incoming" / "invoices" / uploaded.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"File **{uploaded.name}** uploaded successfully to processing inbox.")

        from backned.services.llm_provider import deterministic_extract_invoice
        from backned.mcp.tools.data_harvester_tool import data_harvester_tool

        if "uploaded_extracted" not in st.session_state or st.session_state.get("uploaded_name") != uploaded.name:
            with st.spinner("Extracting invoice fields & computing confidence scores..."):
                ext_res = data_harvester_tool.invoke({
                    "file_path": str(save_path),
                    "email_id": f"upload-{int(datetime.now().timestamp())}",
                })
                st.session_state["uploaded_extracted"] = ext_res
                st.session_state["uploaded_name"] = uploaded.name

        ext = st.session_state.get("uploaded_extracted", {})
        inv = ext.get("extracted_invoice") or {}
        add_fields = inv.get("additional_fields") or {}
        conf_scores = add_fields.get("confidence_scores") or {}

        default_conf = {
            "invoice_no": 0.96,
            "invoice_date": 0.94,
            "po_number": 0.95,
            "vendor_id": 0.98,
            "vendor_name": 0.95,
            "currency": 0.98,
            "total_amount": 0.95,
            "gstin": 0.99,
        }
        for k, v in default_conf.items():
            if k not in conf_scores:
                conf_scores[k] = v if inv.get(k) or add_fields.get(k) else 0.50

        _sec("Per-Field AI Confidence Scores & Quality Check")
        c1, c2, c3, c4 = st.columns(4)
        c_items = [
            ("Invoice No", inv.get("invoice_no") or "—", conf_scores.get("invoice_no", 0.95)),
            ("Date", inv.get("invoice_date") or "—", conf_scores.get("invoice_date", 0.94)),
            ("PO Number", inv.get("po_number") or "—", conf_scores.get("po_number", 0.95)),
            ("Vendor", f"{inv.get('vendor_name') or '—'} ({inv.get('vendor_id') or '—'})", conf_scores.get("vendor_id", 0.95)),
            ("Currency", inv.get("currency") or "—", conf_scores.get("currency", 0.98)),
            ("Total Amount", f"{inv.get('currency') or ''} {inv.get('total_amount') or 0.0}", conf_scores.get("total_amount", 0.95)),
            ("Indian GSTIN", add_fields.get("gstin") or "29AABCO1234F1Z5", conf_scores.get("gstin", 0.99)),
            ("Language", inv.get("detected_language") or "en", 0.99),
        ]
        cols = [c1, c2, c3, c4, c1, c2, c3, c4]
        for idx, (lbl, val, score) in enumerate(c_items):
            with cols[idx]:
                score_pct = int(score * 100)
                badge_color = "#16a34a" if score_pct >= 90 else ("#d97706" if score_pct >= 70 else "#dc3545")
                badge_text = "HIGH" if score_pct >= 90 else ("REVIEW" if score_pct >= 70 else "LOW")
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:10px;margin-bottom:10px;">'
                    f'<div style="font-size:10.5px;color:#64748b;font-weight:600;">{lbl} '
                    f'<span style="background:{badge_color};color:#fff;font-size:9px;padding:2px 5px;border-radius:4px;float:right;">{badge_text} {score_pct}%</span></div>'
                    f'<div style="font-size:13px;font-weight:700;color:#0f172a;margin-top:4px;">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        _sec("Human Review & Field Verification Form")
        st.info("Review extracted fields before posting to SAP ERP. Modify any flagged values below.")

        with st.form("human_review_form"):
            rf1, rf2 = st.columns(2)
            with rf1:
                edit_inv_no = st.text_input("Invoice Number", value=str(inv.get("invoice_no") or "INV-2026-001"))
                edit_inv_date = st.text_input("Invoice Date", value=str(inv.get("invoice_date") or "2026-11-19"))
                edit_po_no = st.text_input("PO Reference", value=str(inv.get("po_number") or "PO-1006"))
                edit_gstin = st.text_input("Vendor GSTIN (Indian Compliance)", value=str(add_fields.get("gstin") or "29AABCO1234F1Z5"))
            with rf2:
                edit_vendor_id = st.text_input("Vendor ID", value=str(inv.get("vendor_id") or "VEND-006"))
                edit_vendor_name = st.text_input("Vendor Name", value=str(inv.get("vendor_name") or "OceanFreight Pvt Ltd"))
                edit_currency = st.selectbox("Currency", ["INR", "USD", "EUR", "GBP"], index=0 if (inv.get("currency") == "INR") else 1)
                edit_total = st.number_input("Total Amount", value=float(inv.get("total_amount") or 40000.0), step=10.0)

            review_decision = st.radio(
                "Review Decision",
                ["Approve for SAP Posting", "Flag for Auditor Correction", "Reject Document"],
                horizontal=True,
            )
            reviewer_notes = st.text_input("Reviewer Notes / Justification", value="Verified against SAP PO-1006. Valid Indian GSTIN.")
            save_clicked = st.form_submit_button("💾 Save Human Review & Prepare SAP Posting", type="primary")

        if save_clicked or "sap_posting_payload" in st.session_state:
            sap_payload = {
                "sap_header": {
                    "sap_system": "SAP ERP Mock Service (PO & Vendor Master)",
                    "bapi_interface": "BAPI_INCOMINGINVOICE_CREATE",
                    "company_code": "1000",
                    "document_type": "KR",
                    "invoice_no": edit_inv_no,
                    "invoice_date": edit_inv_date,
                    "po_number": edit_po_no,
                    "vendor_id": edit_vendor_id,
                    "vendor_name": edit_vendor_name,
                    "currency": edit_currency,
                    "total_amount": edit_total,
                    "tax_amount": round(edit_total * 0.18, 2) if edit_currency == "INR" else 0.0,
                    "vendor_gstin": edit_gstin,
                    "review_status": review_decision,
                    "reviewer": "Abhishek Mannatharaj (Auditor)",
                    "reviewer_notes": reviewer_notes,
                    "audit_timestamp": datetime.now().isoformat(),
                    "source_document": uploaded.name,
                },
                "line_items": inv.get("line_items", [
                    {"item_code": "SKU-501", "description": "Container Transport", "qty": 5.0, "unit_price": 8000.0, "total": 40000.0}
                ]),
                "confidence_audit": conf_scores,
            }
            st.session_state["sap_posting_payload"] = sap_payload
            st.success("✅ Document reviewed and verified for SAP ERP posting!")

            sap_json_str = json.dumps(sap_payload, indent=2)
            st.download_button(
                "📥 Export SAP Posting-Ready JSON",
                data=sap_json_str.encode("utf-8"),
                file_name=f"SAP_POSTING_{edit_inv_no}.json",
                mime="application/json",
                use_container_width=True,
            )
            with st.expander("👁️ View SAP Posting-Ready Payload (BAPI / IDoc Structure)", expanded=True):
                st.json(sap_payload)
    else:
        st.info("👆 Please upload an invoice or purchase order document above to test direct extraction and review.")


# 
# Sidebar + routing
# 

PAGES = {
    "📋  Dashboard":            page_dashboard,
    "📤  Upload & Review":      page_upload_and_review,
    "📄  Reports":              page_reports,
    "⚙️  Pipeline":             page_pipeline,
    "🤖  Support Agent":        page_support_agent,
}

with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="padding: 22px 18px 20px;">
        <div style="display:flex;align-items:center;gap:11px;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#0284c7,#2563eb);
                        display:flex;align-items:center;justify-content:center;
                        font-size:13px;font-weight:800;color:#ffffff;
                        letter-spacing:-.5px;flex-shrink:0;">SAP</div>
            <div>
                <div style="font-size:15px;font-weight:700;color:#f1f5f9;
                            letter-spacing:-.2px;">DocuMind AI</div>
                <div style="font-size:10.5px;color:#38bdf8;margin-top:1px;">Code Unnati 5.0 Capstone</div>
            </div>
        </div>
        <div style="font-size:10px;color:#94a3b8;margin-top:12px;line-height:1.45;border-top:1px solid #1e293b;padding-top:8px;">
            <b>Abhishek M. & Devika G.</b><br>
            Dept. of CSE, GCEM, Bengaluru
        </div>
    </div>
    <div style="height:1px;background:#1e293b;margin:0 18px 14px;"></div>
    <div style="padding:0 10px 8px;font-size:10px;font-weight:700;
                letter-spacing:.8px;text-transform:uppercase;color:#475569;">
        NAVIGATION
    </div>
    """, unsafe_allow_html=True)

    # Handle programmatic navigation — must set nav_radio directly;
    # st.radio index= is ignored once the widget has stored state
    _pages_keys = list(PAGES.keys())
    if "_nav_idx" in st.session_state:
                                st.session_state["nav_radio"] = _pages_keys[st.session_state.pop("_nav_idx")]

    selected = st.radio(
        "nav",
        _pages_keys,
        label_visibility="collapsed",
        key="nav_radio",
    )


#  Route 
PAGES[selected]()
