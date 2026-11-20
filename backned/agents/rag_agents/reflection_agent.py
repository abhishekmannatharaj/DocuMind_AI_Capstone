"""Reflection Agent — evaluates the generated answer for quality and completeness,
flags hallucination risk, and decides whether to accept or request regeneration.

LangGraph flow:  reflect_node → (accept | retry) → END
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any, Literal

from langgraph.graph import StateGraph, END
try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None
from backned.services.llm_provider import call_chat_llm

from backned.config.logging_setup import setup_logging
from backned.config.settings import (
    AZURE_AI_ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT_NAME,
)
from backned.config.observability import observe, langfuse_context

logger = logging.getLogger(__name__)

_MAX_RETRIES = 1   # attempt regeneration at most once

_REFLECTION_SYSTEM = """\
You are a quality-control evaluator for an AI invoice support agent.

Given a user question, the context used, and the generated answer, evaluate:
1. RELEVANCE  — Does the answer directly address the question? (0-10)
2. GROUNDING  — Is the answer supported by the provided context? (0-10)
               Flag if the answer contains facts NOT present in the context.
3. COMPLETENESS — Does the answer cover all key aspects of the question? (0-10)

Respond ONLY with valid JSON in exactly this format:
{
  "relevance": <int 0-10>,
  "grounding": <int 0-10>,
  "completeness": <int 0-10>,
  "overall": <int 0-10>,
  "issues": ["<issue 1>", ...],
  "verdict": "accept" | "retry",
  "feedback": "<one-sentence improvement suggestion if verdict=retry, else empty>"
}

verdict = "accept" if overall >= 6, else "retry".
"""

_client: AzureOpenAI | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=AZURE_AI_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
    return _client


#  State 

class ReflectionState(TypedDict):
    query:         str
    ranked_chunks: list[dict[str, Any]]
    chat_history:  list[dict[str, str]]
    answer:        str
    sources_used:  list[str]
    reflection:    dict[str, Any]
    retry_count:   int
    final_answer:  str
    error:         str


#  Nodes 

def reflect_node(state: ReflectionState) -> ReflectionState:
    """Evaluate the generated answer using LLM-as-judge."""
    logger.info("Reflection Agent: evaluating answer quality ...")

    context_summary = "\n".join(
        f"[{c.get('source','?')}]: {c.get('text','')[:200]}"
        for c in state["ranked_chunks"][:4]
    )

    prompt = (
        f"Question: {state['query']}\n\n"
        f"Context used:\n{context_summary}\n\n"
        f"Generated answer:\n{state['answer']}"
    )

    try:
        messages = [
            {"role": "system", "content": _REFLECTION_SYSTEM},
            {"role": "user",   "content": prompt},
        ]
        llm_resp = call_chat_llm(messages, temperature=0.0, max_tokens=400)
        if llm_resp:
            import json
            content = llm_resp.strip().lstrip("```json").rstrip("```").strip()
            reflection = json.loads(content)
        else:
            reflection = {"verdict": "accept", "overall": 8, "issues": [], "feedback": ""}
    except Exception as exc:
        logger.warning("Reflection parse error: %s - defaulting to accept", exc)
        reflection = {"verdict": "accept", "overall": 7, "issues": [], "feedback": ""}

    logger.info(
        "Reflection scores: overall=%s, verdict=%s",
        reflection.get("overall", "?"), reflection.get("verdict", "?"),
    )
    return {**state, "reflection": reflection}


def accept_node(state: ReflectionState) -> ReflectionState:
    """Accept the answer — pass it through as final."""
    logger.info("Reflection: answer accepted (overall=%s)", state["reflection"].get("overall"))
    return {**state, "final_answer": state["answer"]}


def retry_node(state: ReflectionState) -> ReflectionState:
    """Regenerate the answer incorporating the reflection feedback."""
    feedback = state["reflection"].get("feedback", "")
    logger.info("Reflection: retrying generation (feedback: %s)", feedback)

    from backned.agents.rag_agents.generation_agent import GenerationAgent  # noqa: PLC0415
    gen = GenerationAgent()

    # Append feedback as a system hint via chat_history
    augmented_history = list(state["chat_history"]) + [{
        "role": "system",
        "content": f"[Quality feedback]: {feedback}  Improve the answer accordingly.",
    }]

    result = gen.run(
        query=state["query"],
        ranked_chunks=state["ranked_chunks"],
        chat_history=augmented_history,
    )
    new_answer = result.get("answer", state["answer"])
    return {
        **state,
        "answer":      new_answer,
        "retry_count": state["retry_count"] + 1,
        "final_answer": new_answer,
    }


def _route(state: ReflectionState) -> Literal["accept", "retry"]:
    if state["reflection"].get("verdict") == "retry" and state["retry_count"] < _MAX_RETRIES:
        return "retry"
    return "accept"


#  Graph 

def build_reflection_graph() -> StateGraph:
    g = StateGraph(ReflectionState)
    g.add_node("reflect", reflect_node)
    g.add_node("accept",  accept_node)
    g.add_node("retry",   retry_node)

    g.set_entry_point("reflect")
    g.add_conditional_edges("reflect", _route, {"accept": "accept", "retry": "retry"})
    g.add_edge("accept", END)
    g.add_edge("retry",  END)

    return g.compile()


class ReflectionAgent:
    """Reflection Agent — evaluates and optionally re-generates answers."""

    def __init__(self) -> None:
        self._graph = build_reflection_graph()

    @observe(name="agent-rag-reflection")
    def run(
        self,
        query: str,
        answer: str,
        ranked_chunks: list[dict],
        chat_history: list[dict] | None = None,
        sources_used: list[str] | None = None,
        callbacks: list | None = None,
    ) -> dict[str, Any]:
        """Evaluate the answer and return the final (possibly improved) answer.

        Returns dict with final_answer (str), reflection (dict), retry_count (int).
        """
        state = ReflectionState(
            query=query,
            ranked_chunks=ranked_chunks,
            chat_history=chat_history or [],
            answer=answer,
            sources_used=sources_used or [],
            reflection={},
            retry_count=0,
            final_answer="",
            error="",
        )
        cfg = {"callbacks": callbacks} if callbacks else {}
        final = self._graph.invoke(state, config=cfg)
        return {
            "final_answer": final["final_answer"] or answer,
            "reflection":   final["reflection"],
            "retry_count":  final["retry_count"],
        }
