"""
Main entry point for the hybrid retail analytics agent.
"""
import os
import sys
import json
import logging
from dspy import ChainOfThought, LM, settings
from langgraph.graph import StateGraph

from agent.config import AgentConfig
from agent.dspy_signatures import Router, ExtractConstraints, GenerateSQL, SynthesizeAnswer, AgentState
from agent.graph_hybrid import build_graph
from agent.rag.retrieval import DocumentRetriever
from agent.tools.sqlite_tool import DatabaseInterface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Running hybrid retail analytics agent...")

# ----------------------------
# DSPy & Graph Setup
# ----------------------------
def _setup_dspy(config: AgentConfig):
    """Configure DSPy modules and build the hybrid agent graph"""
    print("🤖 Configuring DSPy modules...")

    # Configure LM FIRST before creating modules
    ollama_model = f"ollama/{config.model_name}"

    os.environ["LITELLM_CACHE"] = "false"
    
    lm = LM(
        model=ollama_model,
        api_base=config.ollama_host,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=120,
        cache=False,
    )
    settings.configure(lm=lm)
   
    print(f"🔧 Configuring LM with model: {ollama_model}")
    print(f"   API base: {config.ollama_host}")
    print(f"   Max tokens: {config.max_tokens}, Temperature: {config.temperature}")
    
    # Test the LM connection
    try:
        print("🧪 Testing LM connection...")
        test_result = lm("Say 'OK' if you can hear me.")
        print(f"✅ LM connection test successful: {test_result}")
    except Exception as e:
        print(f"⚠️  LM connection test failed: {e}")
        print("   Continuing anyway, but responses may be slow...")
    
    print("✅ LM configured")

    # Document retriever
    document_retriever = DocumentRetriever(config)
    db_interface = DatabaseInterface(config)

    # Initialize DSPy modules (ChainOfThought wrappers) AFTER LM is configured
    router_sig = ChainOfThought(Router)
    constraint_extractor_sig = ChainOfThought(ExtractConstraints)
    sql_generator_sig = ChainOfThought(GenerateSQL)
    synthesizer_sig = ChainOfThought(SynthesizeAnswer)

    print("✅ DSPy modules created")

    # Build hybrid agent graph
    graph = build_graph(
        config,
        retriever=document_retriever,
        db_interface=db_interface,
        router_sig=router_sig,
        planner_sig=constraint_extractor_sig,
        sql_sig=sql_generator_sig,
        synth_sig=synthesizer_sig,
    )

    print("✅ Hybrid agent graph compiled")
    return graph

# ----------------------------
# Run agent on a sample question
# ----------------------------
def run_sample_question(graph: StateGraph, question: str, format_hint: str, question_id: str):
    # Initialize agent state
    state: AgentState = AgentState(
        question=question,
        format_hint=format_hint,
        question_id=question_id,
        route="",
        retrievied_chunks=[],
        db_schema="",
        extracted_constraints=[],
        query="",
        query_result={},
        answer="",
        reasoning="",
        citations=[],
        error="",
        repair_attempts=0,
        trace=[],
    )

    # Run the graph with increased recursion limit
    result_state = graph.invoke(state, config={"recursion_limit": 150})

    # Print results
    print("\n--- Agent Answer ---")
    print(result_state["answer"])
    print("\n--- Citations ---")
    print(json.dumps(result_state.get("citations", []), indent=2))
    print("\n--- Trace ---")
    for t in result_state["trace"]:
        print(t)

    return result_state

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        config = AgentConfig()
        graph = _setup_dspy(config)

        # Example: run one sample question
        # sample_question = "According to the product policy, what is the return window (days) for unopened Beverages?"
        # format_hint = "int"
        # question_id = "rag_policy_beverages_return_days"
        sample_question = "During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}."
        format_hint = "{category:str, quantity:int}"
        question_id = "hybrid_top_category_qty_summer_1997"
        # sample_question = "Top 3 products by total revenue all-time. Revenue uses Order Details: SUM(UnitPrice*Quantity*(1-Discount)). Return list[{product:str, revenue:float}]."
        # format_hint = "list[{product:str, revenue:float}]"
        # question_id = "sql_top3_products_by_revenue_alltime"

        run_sample_question(graph, sample_question, format_hint, question_id)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
