"""MCP Clients — Simple tools to connect to the Invoice Auditor server.

This module provides easy ways to fetch tools from the MCP server using SSE.

"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

#  Tool cache (populated on first access) 
_TOOLS_CACHE: list | None = None

#  Server coordinates (must match mcp_server.py HOST / PORT) 
MCP_SERVER_HOST: str = "127.0.0.1"
MCP_SERVER_PORT: int = 8000
MCP_SERVER_URL:  str = f"http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/sse"

# Server name used as the key throughout MultiServerMCPClient
_SERVER_NAME: str = "invoice_auditor"

# Connection map consumed by MultiServerMCPClient
_SERVER_CONFIG: dict[str, dict] = {
    _SERVER_NAME: {
        "url":       MCP_SERVER_URL,
        "transport": "sse",
    },
}


#  Public helpers 

def get_client() -> MultiServerMCPClient:
    """Create a ``MultiServerMCPClient`` pointed at the Invoice Auditor server.

    The client is lightweight and stateless at construction time — no network
    connection is opened until ``get_tools()`` or ``session()`` is called.

    Returns:
        MultiServerMCPClient configured for SSE at ``MCP_SERVER_URL``.

    Example::

        client = get_client()
        tools  = await client.get_tools()
    """
    logger.debug("Initialising MultiServerMCPClient -> %s", MCP_SERVER_URL)
    return MultiServerMCPClient(_SERVER_CONFIG)


async def get_tools() -> list:
    """Fetch all tools registered on the MCP server.

    Opens a transient SSE connection, retrieves every tool exposed by the
    Invoice Auditor server as LangChain ``BaseTool`` objects, and returns
    them immediately.

    Returns:
        List of ``BaseTool`` instances (one per MCP tool).

    Raises:
        httpx.ConnectError: if the server is not reachable.
        Exception: for any other transport or protocol error.

    Example::

        tools = await get_tools()
        print([t.name for t in tools])
    """
    client = get_client()
    tools  = await client.get_tools()
    logger.info(
        "Loaded %d tool(s) from MCP server [%s]",
        len(tools), MCP_SERVER_URL,
    )
    return tools


@asynccontextmanager
async def get_session() -> AsyncGenerator:
    """Async context manager that yields a live MCP ``Session``.

    Use this when you need direct session access (e.g. calling
    ``load_mcp_tools`` manually or invoking low-level MCP primitives).

    Yields:
        An active MCP session for ``_SERVER_NAME``.

    Example::

        async with get_session() as session:
            tools = await load_mcp_tools(session)
    """
    client = get_client()
    async with client.session(_SERVER_NAME) as session:
        logger.info("MCP session open -> %s", MCP_SERVER_URL)
        try:
            yield session
        finally:
            logger.info("MCP session closed")


#  Sync wrappers for agents (non-async code) 

def get_tools_sync(use_cache: bool = True) -> list:
    """Synchronous wrapper for ``get_tools()`` — for use in agents.

    Fetches all tools from the MCP server. Uses a module-level cache
    by default to avoid repeated SSE connections.

    Args:
        use_cache: If True, returns cached tools after first fetch.
                   If False, always fetches fresh tools from server.

    Returns:
        List of LangChain ``BaseTool`` objects.

    Example::

        tools = get_tools_sync()
        invoice_watcher = next(t for t in tools if t.name == "invoice_watcher")
    """
    global _TOOLS_CACHE

    if use_cache and _TOOLS_CACHE is not None:
        logger.debug("Using cached tools (%d tools)", len(_TOOLS_CACHE))
        return _TOOLS_CACHE

    # Fetch tools in a new event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, create a new thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tools = pool.submit(lambda: asyncio.run(get_tools())).result()
    except RuntimeError:
        # No running loop — safe to use asyncio.run directly
        tools = asyncio.run(get_tools())

    if use_cache:
        _TOOLS_CACHE = tools
        logger.debug("Cached %d tools from MCP server", len(tools))

    return tools


def get_tool_by_name(tool_name: str, use_cache: bool = True):
    """Get a specific tool by name from the MCP server.

    Args:
        tool_name: The name of the tool (e.g., "invoice_watcher").
        use_cache: If True, uses cached tools; if False, fetches fresh.

    Returns:
        The requested ``BaseTool`` object.

    Raises:
        ValueError: if the tool name is not found.

    Example::

        tool = get_tool_by_name("data_harvester")
        result = tool.invoke({"file_path": "...", "email_id": "..."})
    """
    tools = get_tools_sync(use_cache=use_cache)
    for tool in tools:
        if tool.name == tool_name:
            return tool
    available = [t.name for t in tools]
    raise ValueError(
        f"Tool '{tool_name}' not found. Available: {', '.join(available)}"
    )


import socket

def _is_mcp_server_running() -> bool:
    """Fast non-blocking check if MCP SSE server is running."""
    try:
        with socket.create_connection((MCP_SERVER_HOST, MCP_SERVER_PORT), timeout=0.2):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _get_local_tool(tool_name: str):
    """Retrieve local in-process tool fallback when MCP SSE server is unreachable."""
    if tool_name == "invoice_watcher":
        from backned.mcp.tools.invoice_watcher_tool import invoice_watcher_tool
        return invoice_watcher_tool
    elif tool_name == "data_harvester":
        from backned.mcp.tools.data_harvester_tool import data_harvester_tool
        return data_harvester_tool
    elif tool_name == "lang_bridge":
        from backned.mcp.tools.lang_bridge_tool import lang_bridge_tool
        return lang_bridge_tool
    elif tool_name == "data_completeness":
        from backned.mcp.tools.data_completeness_checker_tool import data_completeness_checker_tool
        return data_completeness_checker_tool
    elif tool_name == "business_validation":
        from backned.mcp.tools.business_validation_tool import business_validation_tool
        return business_validation_tool
    elif tool_name == "insight_reporter":
        from backned.mcp.tools.insight_reporter_tool import insight_reporter_tool
        return insight_reporter_tool
    elif tool_name == "vector_indexer":
        from backned.mcp.tools.rag_tools.vector_indexer_tool import vector_indexer_tool
        return vector_indexer_tool
    elif tool_name == "semantic_retriever":
        from backned.mcp.tools.rag_tools.semantic_retriever_tool import semantic_retriever_tool
        return semantic_retriever_tool
    elif tool_name == "chunk_ranker":
        from backned.mcp.tools.rag_tools.chunk_ranker_tool import chunk_ranker_tool
        return chunk_ranker_tool
    elif tool_name == "response_synthesizer":
        from backned.mcp.tools.rag_tools.response_synthesizer_tool import response_synthesizer_tool
        return response_synthesizer_tool
    return None


def invoke_tool_sync(tool_name: str, tool_input: dict, use_cache: bool = True) -> dict:
    """Fetch an MCP tool by name and invoke it synchronously, with fast in-process fallback."""
    import json as _json

    local_fallback = _get_local_tool(tool_name)

    # If MCP server is not running, bypass SSE directly to in-process tool
    if not _is_mcp_server_running() and local_fallback is not None:
        res = local_fallback.invoke(tool_input)
        if hasattr(res, "model_dump"):
            return res.model_dump()
        return res
    try:
        tool = get_tool_by_name(tool_name, use_cache=use_cache)
    except Exception as exc:
        if local_fallback is not None:
            logger.info("MCP server unavailable (%s); invoking %s directly in-process", exc, tool_name)
            res = local_fallback.invoke(tool_input)
            if hasattr(res, "model_dump"):
                return res.model_dump()
            return res
        raise

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            raw = pool.submit(lambda: asyncio.run(tool.ainvoke(tool_input))).result()
    except RuntimeError:
        raw = asyncio.run(tool.ainvoke(tool_input))
    except Exception as exc:
        if local_fallback is not None:
            logger.info("MCP invocation failed (%s); fallback to local %s", exc, tool_name)
            res = local_fallback.invoke(tool_input)
            if hasattr(res, "model_dump"):
                return res.model_dump()
            return res
        raise

    # MCP content-item envelope: [{"type": "text", "text": "<json>", ...}]
    # Unwrap and parse so callers always get a plain dict.
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "text":
                try:
                    return _json.loads(item["text"])
                except (_json.JSONDecodeError, KeyError):
                    pass
            elif isinstance(item, str):
                try:
                    return _json.loads(item)
                except _json.JSONDecodeError:
                    pass
        # Couldn't parse — return the list as-is so the caller can inspect it.
        return raw

    if isinstance(raw, str):
        try:
            return _json.loads(raw)
        except _json.JSONDecodeError:
            pass

    return raw


#  Smoke-test 

async def _smoke_test() -> None:  # pragma: no cover
    """Print all tools available on the running MCP server."""
    print(f"\nConnecting to Invoice Auditor MCP Server at {MCP_SERVER_URL} …\n")
    try:
        tools = await get_tools()
        print(f"Found {len(tools)} tool(s):\n")
        for t in tools:
            print(f"• {t.name:<32} {(t.description or '').splitlines()[0][:72]}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n Could not connect: {exc}")
        print(
            "Ensure the server is running:\n"
            ".venv\\Scripts\\python.exe -m backned.mcp.server.mcp_server"
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(_smoke_test())
