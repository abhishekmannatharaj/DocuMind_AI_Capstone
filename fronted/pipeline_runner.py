"""Pipeline Runner — orchestrates all 6 agents end-to-end for the Streamlit UI.

Observability
-------------
Every call to ``run_full_pipeline()`` produces:
  • A top-level LangFuse **trace** named ``invoice-auditor-pipeline``
  • Six child **spans**, one per pipeline stage, each linked to the same trace
  • Automatic LLM generation tracing for every Azure OpenAI call made by the
    tools (via the ``langfuse.openai`` drop-in replacement)
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Callable

#  Ensure project root is on sys.path 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backned.agents.invoice_monitor_agent import InvoiceMonitorAgent
from backned.agents.extractor_agent import ExtractorAgent
from backned.agents.translation_agent import TranslationAgent
from backned.agents.validation_agent import ValidationAgent
from backned.agents.business_validation_agent import BusinessValidationAgent
from backned.agents.reporting_agent import ReportingAgent
from backned.models import MonitorResult

#  LangFuse observability 
from backned.config.observability import (
    get_callback_handler,
    flush_langfuse,
    observe,
    langfuse_context,
)

logger = logging.getLogger(__name__)

# 
# Log capture handler
# 

class _CaptureHandler(logging.Handler):
    """Temporarily captures all log records into a list."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._records: list[str] = []
        self.setFormatter(logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(module)s] %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(self.format(record))

    @property
    def lines(self) -> list[str]:
        return list(self._records)


# 
# Stage metadata (used by the UI)
# 

STAGE_INFO: list[tuple[str, str, str]] = [
    ("", "Monitor",          "Scanning new invoices"),
    ("", "Extract",          "Extracting structured data from PDFs / images"),
    ("", "Translate",        "Translating invoice to English"),
    ("", "Validate",         "Validates missing information based on defined rules"),
    ("", "ERP Validate",     "Validates invoice data against enterprise records"),
    ("", "Report",           "Generate summary reports & recommendations"),
]

STAGE_KEYS = ["monitor", "extract", "translate", "validate", "business_validate", "report"]


# 
# PipelineRunner
# 

class PipelineRunner:
    """Runs the full 6-agent invoice auditing pipeline with optional UI callbacks."""

    @observe(name="invoice-auditor-pipeline")
    def run_full_pipeline(
        self,
        reset: bool = False,
        on_stage_start: Callable[[str, int, int], None] | None = None,
        on_stage_done:  Callable[[str, int, int, list[str]], None] | None = None,
    ) -> dict:
        """Execute all 6 pipeline stages and return a results dict.

        Args:
            reset: Clear processed-email state before running.
            on_stage_start: Called with (stage_key, stage_idx, total).
            on_stage_done:  Called with (stage_key, stage_idx, total, log_lines).

        Returns:
            dict with keys: monitor_result, extraction_results,
            translation_results, validation_results,
            business_validation_results, reporting_summary, all_logs, errors.
        """
        #  Generate a session ID to correlate all spans in LangFuse 
        session_id = f"pipeline-run-{uuid.uuid4().hex[:8]}"

        # Tag the overarching trace with useful metadata
        langfuse_context.update_current_trace(
            name="invoice-auditor-pipeline",
            session_id=session_id,
            tags=["invoice-auditor", "batch-pipeline"],
            metadata={"reset": reset, "stages": STAGE_KEYS},
        )
        trace_id = langfuse_context.get_current_trace_id()
        logger.info("LangFuse trace id: %s (session: %s)", trace_id, session_id)

        results: dict = {"all_logs": {}, "errors": [], "session_id": session_id}
        n = len(STAGE_KEYS)

        def _run(key: str, idx: int, fn: Callable, *args, **kwargs):
            if on_stage_start:
                on_stage_start(key, idx, n)

            # Create a per-stage CallbackHandler linked to the same session
            cb_handler = get_callback_handler(
                name=f"stage-{key}",
                session_id=session_id,
                tags=["invoice-auditor", f"stage-{key}"],
                metadata={"stage_index": idx, "stage_key": key},
            )

            handler = _CaptureHandler()
            root = logging.getLogger()
            root.addHandler(handler)
            try:
                # Inject callbacks kwarg only if the function accepts it
                import inspect
                sig = inspect.signature(fn)
                if "callbacks" in sig.parameters:
                    kwargs["callbacks"] = [cb_handler]
                out = fn(*args, **kwargs)
            except Exception as exc:
                results["errors"].append(f"Stage '{key}' failed: {exc}")
                logger.error("Stage '%s'error: %s", key, exc, exc_info=True)
                out = None
            finally:
                root.removeHandler(handler)
                try:
                    cb_handler.flush()
                except Exception:
                    pass
            results["all_logs"][key] = handler.lines
            if on_stage_done:
                on_stage_done(key, idx, n, handler.lines)
            return out

        #  1. Monitor 
        monitor_agent = InvoiceMonitorAgent()
        if reset:
            monitor_agent.reset_processed_state()

        monitor_result: MonitorResult = _run("monitor", 0, monitor_agent.run) or MonitorResult()
        results["monitor_result"] = monitor_result

        #  2. Extract 
        extraction_results = _run("extract", 1, ExtractorAgent().run, monitor_result) or []
        results["extraction_results"] = extraction_results

        #  3. Translate 
        translation_results = _run("translate", 2, TranslationAgent().run, extraction_results) or []
        results["translation_results"] = translation_results

        #  4. Validate 
        validation_results = _run("validate", 3, ValidationAgent().run, translation_results) or []
        results["validation_results"] = validation_results

        #  5. Business validate 
        biz_results = _run(
            "business_validate", 4,
            BusinessValidationAgent().run, translation_results, validation_results,
        ) or []
        results["business_validation_results"] = biz_results

        #  6. Report 
        reporting_summary = _run(
            "report", 5,
            ReportingAgent().run, translation_results, validation_results, biz_results,
        )
        results["reporting_summary"] = reporting_summary

        #  Record pipeline summary on the LangFuse trace 
        rs = reporting_summary
        langfuse_context.update_current_observation(
            output={
                "total_invoices":   getattr(rs, "total_invoices", 0),
                "auto_approved":    getattr(rs, "auto_approved", 0),
                "manual_review":    getattr(rs, "manual_review", 0),
                "rejected":         getattr(rs, "rejected", 0),
                "errors":           results["errors"],
            },
            metadata={"session_id": session_id},
        )

        # Flush any remaining buffered events to LangFuse
        flush_langfuse()
        return results
