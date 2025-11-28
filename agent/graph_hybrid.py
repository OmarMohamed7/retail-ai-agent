# LangGraph Hybrid Agent Graph

import json
import logging
import sqlite3

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

class RouterNode():
    def __init__(self, router):
        self.logger = logging.getLogger(__name__)
        self.router = router

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("RouterNode: determining route for question")
        question = state["question"].lower()
        
        # First, try keyword-based routing for common patterns
        rag_keywords = ["policy", "according to", "per", "definition", "document", "rule", "return window"]
        sql_keywords = ["total", "sum", "count", "average", "revenue", "sales", "top", "highest", "lowest", "avg"]
        
        has_rag_keywords = any(kw in question for kw in rag_keywords)
        has_sql_keywords = any(kw in question for kw in sql_keywords)
        
        # If question clearly has RAG keywords (especially "policy" or "according to"), use RAG
        if has_rag_keywords and not has_sql_keywords:
            state["route"] = "rag"
            state["trace"].append(f"RouterNode: keyword-based route=rag (keywords: {[kw for kw in rag_keywords if kw in question]})")
            self.logger.info("RouterNode: using keyword-based routing -> rag")
            return state
        
        # Otherwise, use DSPy router
        try:
            result = self.router(query=state["question"])
            route = result.route.lower() if hasattr(result, 'route') else "hybrid"
            
            # Validate route is one of the expected values
            if route not in ["rag", "sql", "hybrid"]:
                self.logger.warning("RouterNode: DSPy returned invalid route '%s', using keyword fallback", route)
                if has_rag_keywords:
                    route = "rag"
                elif has_sql_keywords:
                    route = "sql"
                else:
                    route = "hybrid"
            
            state["route"] = route
            state["trace"].append(f"RouterNode: DSPy route={route}")
            self.logger.info("RouterNode: DSPy returned route=%s", route)
        except Exception as e:
            self.logger.error("RouterNode: DSPy error: %s, using keyword fallback", e)
            # Fallback to keyword-based routing
            if has_rag_keywords:
                state["route"] = "rag"
            elif has_sql_keywords:
                state["route"] = "sql"
            else:
                state["route"] = "hybrid"
            state["trace"].append(f"RouterNode: fallback route={state['route']}")
        
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
            state["extracted_constraints"] = json.loads(constraints_str) if isinstance(constraints_str, str) else constraints_str
        except (json.JSONDecodeError, TypeError):
            state["extracted_constraints"] = []
        state["trace"].append("PlannerNode: extracted constraints")
        return state


class NL2SQLNode():
    def __init__(self, sql_sig, db_interface: DatabaseInterface):
        self.logger = logging.getLogger(__name__)
        self.sql_sig = sql_sig
        self.db_interface = db_interface

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nNL2SQLNode: generating SQL query\n")
        # Convert constraints to string if it's a list
        constraints_str = str(state.get("extracted_constraints", []))
        previous_error_str = state.get("error") or ""
        result = self.sql_sig(
            question=state["question"],
            schema_str=self.db_interface.get_schema(),
            constraints=constraints_str,
            previous_error=previous_error_str,
        )
        state["query"] = result.sql_query
        state["trace"].append(f"NL2SQLNode: sql_query={result.sql_query}")
        return state

class ExecutorNode():
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



class SynthesizerNode():
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
        
        result = self.synth_sig(
            question=state["question"],
            format_hint=state.get("format_hint", ""),
            sql_results=sql_results_str,
            doc_chunks=doc_chunks_str,
        )
        state["answer"] = result.answer
        # Parse citations_json if it's a string, otherwise use as-is
        if isinstance(result.citations_json, str):
            try:
                state["citations"] = json.loads(result.citations_json)
            except json.JSONDecodeError:
                state["citations"] = []
        else:
            state["citations"] = result.citations_json or []
        state["trace"].append("SynthesizerNode: produced answer")
        return state


class RepairLoopNode():
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


class CheckpointerNode():
    """Logs the entire agent state"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self, state: AgentState) -> AgentState:
        self.logger.info("\nCheckpointerNode: logging state trace\n")
        for t in state["trace"]:
            self.logger.info(t)
        return state


class ValidatorNode():
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