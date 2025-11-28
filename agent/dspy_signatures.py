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
    Extract business constraints from retrieved documents.
    
    Given a question and document chunks (as JSON string), extract relevant constraints, dates, or filters.
    Return constraints as a valid JSON array string, e.g., '[{"type": "date_range", "value": "1997-06-01 to 1997-08-31"}]' or '[]' if none.
    DO NOT return document chunks. Only return extracted constraints as JSON.
    """
    question: str = dspy.InputField()
    doc_chunks: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    constraints: str = dspy.OutputField()


class GenerateSQL(dspy.Signature):
    """
    Generate valid SQLite query from natural language question.
    
    Given a question, database schema, constraints (dates, filters), and any previous errors,
    generate a valid SQLite SELECT query.
    
    CRITICAL RULES:
    - Table names with spaces MUST be quoted with double quotes
    - IMPORTANT: The table is called "Order Details" (with a SPACE), NOT "OrderDetails" (no space)
    - Always use "Order Details" (with space and quotes) in SQL, never "OrderDetails"
    - Use table names EXACTLY as shown in schema - DO NOT remove spaces from table names
    - Column names with spaces must also be quoted
    - Use only tables and columns from the provided schema
    - Apply constraints (date ranges, filters) in WHERE clauses
    - For revenue calculations: SUM(UnitPrice * Quantity * (1 - Discount))
    - Return ONLY the SQL query, no explanations or markdown
    - If previous_error is provided, fix the error in the new query
    
    Example: SELECT * FROM "Order Details" (correct) NOT FROM OrderDetails (wrong)
    """
    question: str = dspy.InputField()
    schema: str = dspy.InputField()
    # constraints: str = dspy.InputField()
    # previous_error: str = dspy.InputField()
    # reasoning: str = dspy.OutputField()
    sql_query: str = dspy.OutputField(desc="A valid SQLite SELECT query. No markdown. No commentary. Only SQL.")


class SynthesizeAnswer(dspy.Signature):
    """
    Produce final answer matching the required format.
    
    The format_hint describes the expected output structure:
    - "int" -> return a single integer number
    - "float" -> return a single float number
    - "{key:type, ...}" -> return a JSON object with those keys and types
    - "list[{key:type, ...}]" -> return a JSON array of objects
    
    IMPORTANT: The format_hint describes the STRUCTURE, not literal text to include.
    For example, if format_hint is "list[{product:str, revenue:float}]":
    - Return: [{"product": "Product Name", "revenue": 12345.67}]
    - NOT: [{"product": "Product Name", "revenue:float": "some value"}]
    - The "revenue" key should have an actual float NUMBER value, not the string "revenue:float"
    
    Return a SINGLE JSON object with all three fields: reasoning, answer, and citations_json.
    DO NOT return an array. Return format: {"reasoning": "...", "answer": "...", "citations_json": "..."}
    The citations_json should be a JSON string containing an array of citation objects.
    The answer field should contain the actual formatted answer matching the format_hint structure.
    """
    question: str = dspy.InputField()
    format_hint: str = dspy.InputField()
    sql_results: str = dspy.InputField()
    doc_chunks: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()
    citations_json: str = dspy.OutputField()
