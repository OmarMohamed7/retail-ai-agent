# DSPy Signatures/Modules

from ast import Dict
from typing import Any, Literal, TypedDict


class AgentState(TypedDict):
    """Agent state."""

    # Input
    question: str
    format_hint: str
    question_id: str

    # Routing
    route: Literal["rag", "sql", "hybrid"]

    # RAG Component
    retrievied_chunks: list[Dict[str, Any]]

    # SQL Component
    db_schema: str
    extracted_constraints: list[Dict[str, Any]]
    query: str
    query_result: Dict[str, Any]  # {"success": bool, "error": str, "result": Any}

    # Synthesis
    answer: str
    citations: list[Dict[str, Any]]  # {"type": table|document, "source": str}

    # Control
    error: str
    repair_attempts: int
    trace: list[str]
