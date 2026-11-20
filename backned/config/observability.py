"""Observability — central LangFuse setup for real-time tracing, monitoring,
and debugging of all agentic AI workflows in Invoice Auditor.

Architecture (LangFuse SDK v3)
------------------------------
Tracing is achieved through two complementary mechanisms:

1. **@observe decorator** (primary) — wraps every agent ``run()`` method and
   the top-level pipeline/RAG orchestrators with a LangFuse span.  Spans are
   automatically nested: the outer ``@observe``-d function creates a *trace*
   and every inner ``@observe``-d call becomes a *generation* or *span* child.

2. **langfuse.openai drop-in** — every file that calls ``AzureOpenAI``
   imports from ``langfuse.openai`` instead of ``openai``.  This patches the
   client in-place so that *every* chat-completion is automatically captured
   as a LangFuse generation with model name, token counts, and latency.

Usage quick-reference
---------------------
    # Decorate any function to create a span
    from backned.config.observability import observe
    @observe(name="my-agent-run")
    def run(self, ...): ...

    # Add rich metadata inside an @observe-d function
    from backned.config.observability import langfuse_context
    langfuse_context.update_current_observation(
        input={"query": ...},
        output={"answer": ...},
        metadata={"latency_ms": 250},
    )
    langfuse_context.update_current_trace(
        tags=["rag", "invoice-auditor"],
        session_id="run-abc123",
    )

    # Low-level Langfuse client for manual trace / span / score creation
    from backned.config.observability import get_langfuse_client
    lf = get_langfuse_client()
    trace = lf.trace(name="custom-trace")

    # Flush all pending events before exit
    from backned.config.observability import flush_langfuse
    flush_langfuse()
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#  Bootstrap env vars before langfuse initialises 
# langfuse.decorators.observe reads LANGFUSE_* from the environment lazily on
# first call.  We ensure load_dotenv() has run by importing settings first.
try:
    from backned.config import settings as _s  # noqa: F401 — side-effect: load_dotenv
except Exception:
    pass

#  Lazy import guard 
# LangFuse is an optional dependency; if it's missing we fall back to no-op
# shims so the rest of the application keeps working unchanged.

try:
    from langfuse import Langfuse, observe as _lf_observe, get_client as _lf_get_client
    _LANGFUSE_AVAILABLE = True

    #  LangFuse 3.x "context" shim 
    # In LangFuse 3.x there is no `langfuse.decorators.langfuse_context`;
    # the equivalent methods are on the shared Langfuse client instance.
    class _LangfuseContext:
        """Thin compatibility shim mapping v2-style langfuse_context calls to
        the LangFuse 3.x ``get_client()`` singleton."""

        def update_current_observation(self, **kw) -> None:
            try:
                client = _lf_get_client()
                # Map v2 kwarg names → v3 method
                if "output" in kw or "input" in kw or "metadata" in kw or "model" in kw:
                    client.update_current_span(**{
                        k: v for k, v in kw.items()
                        if k in ("input", "output", "metadata", "model",
                                 "usage", "usage_details", "status_message",
                                 "version", "level")
                    })
            except Exception:
                pass

        def update_current_trace(self, **kw) -> None:
            try:
                _lf_get_client().update_current_trace(**kw)
            except Exception:
                pass

        def get_current_trace_id(self) -> str | None:
            try:
                return _lf_get_client().get_current_trace_id()
            except Exception:
                return None

        def get_current_observation_id(self) -> str | None:
            try:
                return _lf_get_client().get_current_observation_id()
            except Exception:
                return None

        def flush(self) -> None:
            try:
                _lf_get_client().flush()
            except Exception:
                pass

    langfuse_context = _LangfuseContext()

    def observe(fn=None, *, name=None, as_type=None, **kw):
        """Thin wrapper that forwards to langfuse.observe with the correct v3 API."""
        if fn is not None:
            return _lf_observe(fn, name=name, as_type=as_type, **kw)
        def _decorator(f):
            return _lf_observe(f, name=name, as_type=as_type, **kw)
        return _decorator

except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning(
        "langfuse package not found — observability disabled.  "
        "Run `pip install langfuse` to enable tracing."
    )

    #  No-op shims 
    class Langfuse:  # type: ignore[no-redef]
        def __init__(self, **kw): pass
        def trace(self, **kw): return _NoOpSpan()
        def flush(self): pass

    def observe(fn=None, *, name=None, **kw):  # type: ignore[misc]
        """No-op @observe when langfuse is unavailable."""
        if fn is not None:
            return fn
        def _decorator(f):
            return f
        return _decorator

    class _NoOpContext:
        def update_current_observation(self, **kw): pass
        def update_current_trace(self, **kw): pass
        def get_current_trace_id(self): return None
        def get_current_observation_id(self): return None
        def flush(self): pass

    langfuse_context = _NoOpContext()  # type: ignore[assignment]

    class _NoOpSpan:
        def span(self, **kw): return self
        def generation(self, **kw): return self
        def event(self, **kw): return self
        def score(self, **kw): return self
        def update(self, **kw): return self
        def end(self, **kw): return self
        def flush(self): pass


#  No-op CallbackHandler stub 
# langfuse.langchain.CallbackHandler requires the full `langchain` package.
# We inherit from langchain-core's BaseCallbackHandler which provides default
# no-op implementations for every callback method that LangGraph invokes.
# When `langchain` is installed, swap this for:
#   from langfuse.langchain import CallbackHandler

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseCBH

    class CallbackHandler(_BaseCBH):
        """No-op CallbackHandler — active tracing is done via @observe decorators.

        Inherits all default empty implementations from langchain-core so that
        passing an instance to ``graph.invoke(config={"callbacks": [...]})``
        is safe and silent.

        To enable full LangGraph tool-level tracing, install the full
        ``langchain`` package and replace with
        ``from langfuse.langchain import CallbackHandler``.
        """
        def __init__(self, **kw):
            super().__init__()

        def flush(self) -> None:
            pass

except ImportError:

    class CallbackHandler:  # type: ignore[no-redef]
        """Minimal no-op stub for when even langchain-core is absent."""
        def __init__(self, **kw): pass
        def flush(self): pass


#  Singleton LangFuse client 

@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse:
    """Return the module-level singleton Langfuse client.

    Credentials are read from environment variables (via settings.py):
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST          (default: https://cloud.langfuse.com)
    - LANGFUSE_ENABLED       (default: true)
    """
    try:
        from backned.config.settings import (
            LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
            LANGFUSE_HOST, LANGFUSE_ENABLED,
        )
    except Exception:
        return Langfuse()

    if not _LANGFUSE_AVAILABLE or not LANGFUSE_ENABLED:
        logger.info("LangFuse observability DISABLED.")
        return Langfuse()

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
            "tracing will be skipped.  Add them to your .env file."
        )

    # Set env vars so the @observe singleton client picks them up
    # (langfuse.get_client() also reads from env vars at initialisation time)
    import os
    if LANGFUSE_PUBLIC_KEY:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY",  LANGFUSE_PUBLIC_KEY)
    if LANGFUSE_SECRET_KEY:
        os.environ.setdefault("LANGFUSE_SECRET_KEY",  LANGFUSE_SECRET_KEY)
    if LANGFUSE_HOST:
        os.environ.setdefault("LANGFUSE_HOST",         LANGFUSE_HOST)

    logger.info("LangFuse observability ENABLED - host: %s", LANGFUSE_HOST)
    return Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
    )


#  Callback handler factory (no-op stub) 

def get_callback_handler(
    name: str = "invoice-auditor",
    *,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> CallbackHandler:
    """Return a ``CallbackHandler`` for LangGraph / LangChain tracing.

    Currently returns a no-op stub because ``langchain`` (the full package,
    not just ``langchain-core``) is not installed.  With ``langchain``
    installed the real ``langfuse.langchain.CallbackHandler`` is used instead.

    Primary tracing is provided via the ``@observe`` decorator applied to
    every agent ``run()`` method and the top-level pipeline orchestrators.
    """
    return CallbackHandler()


#  Flush helper 

def flush_langfuse() -> None:
    """Flush all pending LangFuse events.  Call this before process exit or
    between pipeline runs to guarantee delivery."""
    try:
        get_langfuse_client().flush()
    except Exception as exc:
        logger.debug("LangFuse flush (client) warning: %s", exc)
    # Also flush the singleton used by @observe
    if _LANGFUSE_AVAILABLE:
        try:
            _lf_get_client().flush()
        except Exception as exc:
            logger.debug("LangFuse flush (observe client) warning: %s", exc)


#  Convenience re-exports 

__all__ = [
    "get_langfuse_client",
    "get_callback_handler",
    "flush_langfuse",
    "observe",
    "langfuse_context",
    "CallbackHandler",
]

