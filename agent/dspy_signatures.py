"""DSPy Signatures/Modules"""

from typing import Any, Dict, List, TypedDict

import dspy

class AgentState(TypedDict):
    """Agent state."""

    # Input
    question: str
    format_hint: str
    question_id: str

    # Routing
    route: str

    reasoning: str

    # RAG Component
    retrievied_chunks: List[Dict[str, Any]]

    # SQL Component
    db_schema: str
    extracted_constraints: List[Dict[str, Any]]
    query: str
    query_result: Dict[str, Any]  # {"success": bool, "error": str, "result": Any}

    # Synthesis
    answer: str
    citations: List[Dict[str, Any]]  # {"type": table|document, "source": str}

    # Control
    error: str
    repair_attempts: int
    trace: List[str]


class Router(dspy.Signature):
    """Router.
    
    Route the agent to the appropriate component based on the query.
    
    Use "rag" when:
    - Question asks about policies, definitions, rules, or documentation
    - Question contains words like: policy, definition, according to, per, document, rule
    - Example: "According to the product policy, what is the return window for unopened Beverages?" -> "rag"
    - Example: "What is the AOV definition?" -> "rag"
    
    Use "sql" when:
    - Question asks for calculations, aggregations, or data from database tables
    - Question contains words like: total, sum, count, average, revenue, sales, top, highest
    - Example: "What is the total revenue for Beverages?" -> "sql"
    - Example: "Top 3 products by revenue" -> "sql"
    
    Use "hybrid" when:
    - Question needs both document information AND database queries
    - Example: "Total revenue from Beverages during Summer Beverages 1997 (from marketing calendar)" -> "hybrid"
    """

    query: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    route: str = dspy.OutputField()


class ExtractConstraints(dspy.Signature):
    """
    Extract business constraints from retrieved documents
    """
    question: str = dspy.InputField()
    doc_chunks: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    constraints: str = dspy.OutputField()


class GenerateSQL(dspy.Signature):
    """
    Generate valid SQLite query from natural language
    """
    question: str = dspy.InputField()
    schema_str: str = dspy.InputField()
    constraints: str = dspy.InputField()
    previous_error: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    sql_query: str = dspy.OutputField()


class SynthesizeAnswer(dspy.Signature):
    """
    Produce final answer matching the required format
    """
    question: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    sql_results: str = dspy.InputField()
    doc_chunks: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()
    citations_json: str = dspy.OutputField()
