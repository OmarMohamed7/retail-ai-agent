# LangGraph Hybrid Agent Graph

import json
import logging
import sqlite3
import re

from langgraph.graph import END, StateGraph

from agent.dspy_signatures import (
    Router,
    ExtractConstraints,
    GenerateSQL,
    SynthesizeAnswer,
    AgentState,
)
from agent.rag.retrieval import DocumentRetriever
from agent.config import AgentConfig
from agent.tools.sqlite_tool import DatabaseInterface




# ----------------------------
# Node Wrappers
# --
class RouterNode:
    """ Router Determine which route the question will be classified as: rag, sql, or hybrid """
    def __init__(self, router):
        self.router = router
        self.logger = logging.getLogger(__name__)

    def run(self, state: AgentState) -> AgentState:
        q = state["question"].lower()
        self.logger.info("RouterNode: routing question")

        # Detect document/definition intent
        rag_signals = [
            "according to", "as defined", "definition", "policy",
            "per the", "document", "kpi docs", "marketing calendar"
        ]
        
        # Detect numeric/aggregation intent
        sql_signals = [
            "total", "sum", "count", "top", "highest", "lowest",
            "revenue", "quantity", "avg", "average", "unitprice", "discount"
        ]

        has_rag = any(k in q for k in rag_signals)
        has_sql = any(k in q for k in sql_signals)

        # --- Routing logic ---
        
        # 1. Hybrid: doc reference + SQL metric
        if has_rag and has_sql:
            route = "hybrid"
            state["route"] = route
            state["trace"].append("RouterNode: hybrid (doc reference + SQL metric)")
            return state

        # 2. RAG only
        if has_rag:
            route = "rag"
            state["route"] = route
            state["trace"].append("RouterNode: rag (doc/definition lookup)")
            return state

        # 3. SQL only
        if has_sql:
            route = "sql"
            state["route"] = route
            state["trace"].append("RouterNode: sql (aggregation metric)")
            return state

        # 4. Fallback to DSPy router
        try:
            result = self.router(query=state["question"])
            route = getattr(result, "route", "hybrid").lower()
            if route not in ["rag", "sql", "hybrid"]:
                route = "hybrid"
            state["route"] = route
            state["trace"].append(f"RouterNode: dsp router={route}")
        except Exception as e:
            self.logger.error("RouterNode: DSPy error: %s", e)
            state["route"] = "hybrid"
            state["trace"].append("RouterNode: fallback=hybrid")

        return state



class RetrieverNode():
    def __init__(self, retriever: DocumentRetriever):
        self.logger = logging.getLogger(__name__)
        self.retriever = retriever

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nRetrieverNode: retrieving top-k chunks\n")
        state["retrievied_chunks"] = self.retriever.retrieve(state["question"])
        state["trace"].append(f"RetrieverNode: retrieved {len(state['retrievied_chunks'])} chunks")
        return state

class PlannerNode():
    def __init__(self, planner_sig):
        self.logger = logging.getLogger(__name__)
        self.planner_sig = planner_sig

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nPlannerNode: extracting constraints from chunks\n")
        # Convert doc_chunks to JSON string for DSPy
        doc_chunks = state.get("retrievied_chunks", [])
        doc_chunks_str = json.dumps(doc_chunks) if doc_chunks else "[]"
        # Call DSPy signature
        result = self.planner_sig(
            question=state["question"],
            doc_chunks=doc_chunks_str,
        )
        # Parse constraints if it's a string, otherwise use as-is
        constraints_str = result.constraints
        try:
            parsed = self.extract_json(constraints_str)
            state["extracted_constraints"] = parsed if isinstance(parsed, (list, dict)) else []
        except (json.JSONDecodeError, TypeError):
            self.logger.error("PlannerNode: failed to parse constraints: %s", constraints_str)
            state["extracted_constraints"] = []
        state["trace"].append("PlannerNode: extracted constraints")
        return state

    def extract_json(self, text: str):
        """
        Extracts the first valid JSON object or array from an LLM response.
        Fixes common malformed JSON patterns.
        """
        if not text:
            return None

        # 1. Remove trailing explanations after JSON
        # Keep only first {...} or [...]
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if not json_match:
            return None

        json_str = json_match.group(1).strip()

        # 2. Fix common LLM mistake: ["key": value] → {"key": value}
        if json_str.startswith("[") and ":" in json_str and not json_str.strip().startswith("[{"):
            json_str = "{" + json_str.lstrip("[").rstrip("]") + "}"

        # 3. Try to load strict JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 4. Try a relaxed repair attempt
        repaired = (
            json_str
            .replace("'", '"')             # fix quotes
            .replace(",}", "}")            # trailing comma
            .replace(",]", "]")
        )

        try:
            return json.loads(repaired)
        except:
            return None


class NL2SQLNode:
    def __init__(self, sql_sig, db_interface: DatabaseInterface):
        self.logger = logging.getLogger(__name__)
        self.sql_sig = sql_sig
        self.db_interface: DatabaseInterface = db_interface

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nNL2SQLNode: generating SQL query\n")
        
        prompt = self.generate_sql_prompt(state["question"], state=state)

        result = self.sql_sig(
            question= prompt,
            schema= self.db_interface.schema_cache
            
        )
        if isinstance(result, str):
            sql_query = result.strip()
        elif hasattr(result, "sql_query"):
            sql_query = result.sql_query.strip()
        else:
            sql_query = str(result).strip()

        if sql_query.startswith("```sql"):
            sql_query = sql_query.split("```sql", 1)[1].split("```")[0].strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.split("```", 1)[1].split("```")[0].strip()

        state["query"] = sql_query
        state["trace"].append(f"NL2SQLNode: sql_query={state['query']}")
        return state

    def generate_sql_prompt(self, question: str, state: AgentState) -> str:
        """Generate the SQL query prompt"""
        constraints = str(state.get("extracted_constraints", []))
        previous_error = state.get("error") or ""

        tables = ", ".join(self.db_interface.tables)
        schema = self.db_interface.schema_cache

        prompt = (
            f"Question: {question}\n"
            f"Database tables: {tables}\n"
            f"Database schema: {schema}\n"
            f"Constraints: {constraints}\n"
            f"Previous error: {previous_error}\n\n"

            "You are an expert SQLite3 engineer. Generate a SINGLE-LINE, VALID, EXECUTABLE SQLite query that answers the question.\n\n"

            "CRITICAL RULES:\n"
            "1. OUTPUT ONLY THE SQL QUERY — no explanations, no markdown, no comments.\n"
            "2. The query MUST be ONE LINE (no \\n).\n"
            "3. Use ONLY tables and columns related to the table not columns from other table. from the schema. DO NOT invent columns.\n"
            "4. DATE RULES:\n"
            "   - If the question includes an explicit date range (e.g., 2012–2014), use direct text comparison:\n"
            "     OrderDate >= 'YYYY-MM-DD' AND OrderDate < 'YYYY-MM-DD'.\n"
            "   - If filtering by year only, use: strftime('%Y', OrderDate) = 'YYYY'.\n"
            "   - If filtering by month only, use: strftime('%m', OrderDate) = 'MM'.\n"
            "   - DO NOT compare strftime('%m') or strftime('%Y') against full timestamps.\n"
            "   - DO NOT use YEAR() or MONTH() — SQLite does not support them.\n"
            "5. When joining: follow foreign keys exactly (e.g., 'Order Details'.OrderID → Orders.OrderID).\n"
            "6. DO NOT use SELECT *. Always list columns explicitly.\n"
            "7. Use double quotes ONLY for names containing spaces (e.g., 'Order Details').\n"
            "8. NO trailing commas.\n"
            "9. The query must end with a semicolon.\n"
            "10. Start directly with SELECT.\n\n"
            "11. Do NOT generate aliases containing braces {}. Use plain SQL identifiers (e.g., AS category).\n"
            "12. ORDER BY must reference an explicitly selected alias or column name from the SELECT clause.\n"
            "13. NEVER invent alias names that do not appear in the SELECT list.\n"
            "14. If the query selects CategoryName, you MUST join both Products (p) AND Categories (c):\n"
            "   JOIN Products AS p ON od.ProductID = p.ProductID\n"
            "   JOIN Categories AS c ON p.CategoryID = c.CategoryID\n"
            "   Do NOT select CategoryName unless Categories is joined.\n"


            "EXAMPLES:\n"
            "SELECT COUNT(*) FROM Customers WHERE Country = 'USA';\n"
            "SELECT p.ProductName, SUM(od.Quantity) FROM Products AS p JOIN 'Order Details' AS od ON p.ProductID = od.ProductID JOIN Orders AS o ON od.OrderID = o.OrderID WHERE strftime('%Y', o.OrderDate) = '1997' AND strftime('%m', o.OrderDate) = '07' GROUP BY p.ProductName;\n"
            "SELECT c.CustomerID, SUM(od.UnitPrice * od.Quantity - (od.Discount * od.UnitPrice * od.Quantity)) AS margin FROM 'Order Details' od JOIN Orders o ON od.OrderID = o.OrderID JOIN Customers c ON o.CustomerID = c.CustomerID WHERE o.OrderDate >= '2012-01-01' AND o.OrderDate < '2014-01-01' GROUP BY c.CustomerID ORDER BY margin DESC LIMIT 1;\n\n"

            "Now generate the SQL query. OUTPUT ONLY THE SQL:\n"
        )
        return prompt


class ExecutorNode:
    """Executes the SQL query and returns the result"""
    def __init__(self, db_path: str):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nExecutorNode: executing SQL query\n")
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(state["query"])
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
            state["query_result"] = {"success": True, "result": rows, "columns": columns, "error": None}
        except (sqlite3.Error, sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
            state["query_result"] = {"success": False, "result": None, "columns": [], "error": str(e)}
            state["error"] = str(e)
        state["trace"].append(f"ExecutorNode: executed SQL, success={state['query_result']['success']}")
        return state



class SynthesizerNode:
    """Produces the final answer"""
    def __init__(self, synth_sig):
        self.logger = logging.getLogger(__name__)
        self.synth_sig = synth_sig

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nSynthesizerNode: producing final answer\n")
        # Convert complex types to JSON strings for DSPy
        sql_results = state.get("query_result") or {}
        sql_results_str = json.dumps(sql_results)
        doc_chunks = state.get("retrievied_chunks", [])
        doc_chunks_str = json.dumps(doc_chunks)
        
        try:
            result = self.synth_sig(
                question=state["question"],
                format_hint=state.get("format_hint", ""),
                sql_results=sql_results_str,
                doc_chunks=doc_chunks_str,
            )
            
            # Handle result - could be object with attributes or dict
            if hasattr(result, 'answer'):
                state["answer"] = result.answer
                citations_json = getattr(result, 'citations_json', '[]')
            elif isinstance(result, dict):
                state["answer"] = result.get("answer", "")
                citations_json = result.get("citations_json", "[]")
            else:
                state["answer"] = getattr(result, 'answer', '')
                citations_json = getattr(result, 'citations_json', '[]')
            
            # Parse citations_json if it's a string, otherwise use as-is
            if isinstance(citations_json, str):
                try:
                    state["citations"] = json.loads(citations_json)
                except json.JSONDecodeError:
                    state["citations"] = []
            else:
                state["citations"] = citations_json or []
        except Exception as e:
            self.logger.error("SynthesizerNode error: %s", e)
          
            error_str = str(e)
            state["answer"] = ""
            state["citations"] = []
            
            try:
                # Look for JSON array in error message (LM returned array instead of object)
                # Match balanced brackets to find complete JSON array
                bracket_count = 0
                start_idx = error_str.find('[')
                if start_idx != -1:
                    for i in range(start_idx, len(error_str)):
                        if error_str[i] == '[':
                            bracket_count += 1
                        elif error_str[i] == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                # Found complete array
                                json_str = error_str[start_idx:i+1]
                                json_data = json.loads(json_str)
                                # Combine fields from array of objects
                                combined = {}
                                for item in json_data:
                                    if isinstance(item, dict):
                                        combined.update(item)
                                if "answer" in combined:
                                    state["answer"] = combined["answer"]
                                if "citations_json" in combined:
                                    try:
                                        state["citations"] = json.loads(combined["citations_json"])
                                    except (json.JSONDecodeError, TypeError):
                                        state["citations"] = []
                                self.logger.info("SynthesizerNode: Extracted answer from malformed JSON array")
                                break
            except (json.JSONDecodeError, ValueError, AttributeError) as parse_error:
                self.logger.warning("SynthesizerNode: Failed to parse error response: %s", parse_error)
        
        state["trace"].append("SynthesizerNode: produced answer")
        return state


class RepairLoopNode:
    """Logs repair attempt - actual retry is handled by graph cycle"""
    def __init__(self, nl2sql_node: NL2SQLNode, executor_node: ExecutorNode, synth_node: SynthesizerNode):
        self.logger = logging.getLogger(__name__)
        # Store nodes for potential future use, but graph handles the cycle
        self.nl2sql_node = nl2sql_node
        self.executor_node = executor_node
        self.synth_node = synth_node

    def run(self, state: AgentState) -> AgentState:
        repair_attempts = state.get("repair_attempts", 0)
        self.logger.info("\nRepairLoopNode: repair attempt %d\n", repair_attempts)
        state["trace"].append(f"RepairLoopNode: repair attempt {repair_attempts}")
        return state


class CheckpointerNode:
    """Logs the entire agent state"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nCheckpointerNode: logging state trace\n")
        for t in state["trace"]:
            self.logger.info(t)
        return state


class ValidatorNode:
    """Checks if the answer is valid"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nValidatorNode: checking if answer is valid\n")
        if state["answer"]:
            return state
        else:
            return state


# ----------------------------
# Build Graph
# ----------------------------
def build_graph(config: AgentConfig,
                retriever: DocumentRetriever,
                db_interface: DatabaseInterface,
                router_sig: Router,
                planner_sig: ExtractConstraints,
                sql_sig: GenerateSQL,
                synth_sig: SynthesizeAnswer) -> StateGraph:
    """Build LangGraph workflow"""

    workflow = StateGraph(AgentState)

    # Instantiate nodes with required dependencies
    router_node = RouterNode(router_sig)
    retriever_node = RetrieverNode(retriever)
    planner_node = PlannerNode(planner_sig)
    nl2sql_node = NL2SQLNode(sql_sig, db_interface=db_interface)
    executor_node = ExecutorNode(config.db_path)
    synth_node = SynthesizerNode(synth_sig)
    repair_node = RepairLoopNode(nl2sql_node, executor_node, synth_node)
    checkpoint_node = CheckpointerNode()

    # Create callable wrapper functions for LangGraph
    def router_func(state: AgentState) -> AgentState:
        return router_node.run(state)
    
    def retriever_func(state: AgentState) -> AgentState:
        return retriever_node.run(state)
    
    def planner_func(state: AgentState) -> AgentState:
        return planner_node.run(state)
    
    def nl2sql_func(state: AgentState) -> AgentState:
        return nl2sql_node.run(state)
    
    def executor_func(state: AgentState) -> AgentState:
        return executor_node.run(state)
    
    def synth_func(state: AgentState) -> AgentState:
        return synth_node.run(state)
    
    def repair_func(state: AgentState) -> AgentState:
        # Increment repair attempts counter
        state["repair_attempts"] = state.get("repair_attempts", 0) + 1
        return repair_node.run(state)
    
    def checkpoint_func(state: AgentState) -> AgentState:
        return checkpoint_node.run(state)

    # Add nodes
    workflow.add_node("router", router_func)
    workflow.add_node("retrieve", retriever_func)
    workflow.add_node("extract_constraints", planner_func)
    workflow.add_node("generate_sql", nl2sql_func)
    workflow.add_node("execute_sql", executor_func)
    workflow.add_node("synthesize", synth_func)
    workflow.add_node("repair", repair_func)
    workflow.add_node("checkpoint", checkpoint_func)

    # Entry point
    workflow.set_entry_point("router")

    # Router branches
    workflow.add_conditional_edges(
        "router",
        lambda state: state['route'],
        {
            "rag": "retrieve",
            "sql": "generate_sql",
            "hybrid": "retrieve"
        }
    )

    # After retrieve
    workflow.add_edge("retrieve", "extract_constraints")

    # After extract constraints
    workflow.add_conditional_edges(
        "extract_constraints",
        lambda state: "generate_sql" if state['route'] in ["sql", "hybrid"] else "synthesize",
        {
            "generate_sql": "generate_sql",
            "synthesize": "synthesize"
        }
    )

    # After SQL generation
    workflow.add_edge("generate_sql", "execute_sql")

    # After execution
    def execution_check(state):
        query_result = state.get("query_result", {})
        repair_attempts = state.get("repair_attempts", 0)
        max_repairs = config.max_repairs
        
        if query_result.get("success"):
            return "synthesize"
        elif repair_attempts >= max_repairs:
            # Max repairs reached, synthesize anyway (even if query failed)
            return "synthesize"
        else:
            return "repair"

    workflow.add_conditional_edges(
        "execute_sql",
        execution_check,
        {
            "synthesize": "synthesize",
            "repair": "repair"
        }
    )

    # Repair loop
    workflow.add_edge("repair", "generate_sql")

    # End / checkpoint
    workflow.add_edge("synthesize", "checkpoint")
    workflow.add_edge("checkpoint", END)

    return workflow.compile()