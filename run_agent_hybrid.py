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
import argparse


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

    
    lm = LM(
        model=ollama_model,
        api_base=config.ollama_host,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=120,
        cache=False,
    )
    settings.configure(lm=lm)
    
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
# Confidence Calculation
# ----------------------------
def calculate_confidence(result_state: AgentState) -> float:
    """
    Calculate confidence score (0.0 to 1.0) based on:
    - Answer quality (non-empty, not "None", not "{}")
    - SQL execution success
    - Presence of citations
    - Query result quality
    """
    confidence = 0.0
    
    # Base score for having an answer
    answer = result_state.get("answer", "").strip()
    if not answer or answer.lower() in ["none", "null", "{}", "[]", ""]:
        return 0.0  # No valid answer = 0 confidence
    
    confidence += 0.3  # Base score for having a non-empty answer
    
    # Check SQL execution if SQL was used
    query = result_state.get("query", "").strip()
    if query:
        query_result = result_state.get("query_result", {})
        if query_result.get("success", False):
            confidence += 0.4  # SQL executed successfully
            # Check if we got actual results
            rows = query_result.get("result") or query_result.get("rows", [])
            if rows and len(rows) > 0:
                confidence += 0.2  # Got actual data from SQL
        else:
            # SQL failed - reduce confidence
            confidence -= 0.2
    else:
        # RAG-only route - check citations
        citations = result_state.get("citations", [])
        if citations and len(citations) > 0:
            confidence += 0.3  # Have citations from documents
    
    # Check if answer is well-formed JSON (for structured outputs)
    format_hint = result_state.get("format_hint", "")
    if format_hint and ("{" in format_hint or "[" in format_hint):
        try:
            parsed = json.loads(answer)
            if isinstance(parsed, (dict, list)) and len(parsed) > 0:
                confidence += 0.1  # Well-formed structured output
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON, that's okay for simple types
    
    # Check for errors
    error = result_state.get("error", "")
    if error:
        confidence -= 0.2  # Had errors during execution
    
    # Normalize to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))
    
    return round(confidence, 2)

# ----------------------------
# Run agent
# ----------------------------
def run_agent(graph: StateGraph, question: str, format_hint: str, question_id: str):
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
        parser = argparse.ArgumentParser(
            description="Retail Analytics Agent - RAG + SQL Hybrid System"
        )
        parser.add_argument("--batch", required=True, help="Path to input JSONL file with questions")
        parser.add_argument("--out", required=True, help="Path to output JSONL file for results")
        parser.add_argument("--verbose", action="store_true", help="Show detailed trace")
        args = parser.parse_args()

        print("=" * 70)
        print("        RETAIL ANALYTICS AGENT - Hybrid RAG + SQL System")
        print("=" * 70)
        print()

        config = AgentConfig()
        graph = _setup_dspy(config)

        print(f"📖 Loading questions from {args.batch}")
        questions = []
        with open(args.batch, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        print(f"✅ Loaded {len(questions)} questions\n")
        print("=" * 70)

        results = []

        for i, q in enumerate(questions, 1):
            print(f"\n🔍 Processing Question {i}/{len(questions)}")
            print(f"   ID: {q['id']}")
            print(f"   Question: {q['question']}")
            print(f"   Format Hint: {q['format_hint']}")
            print("=" * 70)

            # Run the agent
            result_state = run_agent(
                graph,
                q['question'],
                q['format_hint'],
                q['id']
            )

            # Calculate confidence based on answer quality
            confidence = calculate_confidence(result_state)
            
            # Build output according to Output Contract
            output_line = {
                "id": q['id'],
                "final_answer": result_state.get("answer", ""),  # matches format_hint
                "sql": result_state.get("query", ""),  # last executed SQL, empty if RAG-only
                "confidence": confidence,
                "explanation": "Generated by hybrid agent in ≤ 2 sentences.",
                "citations": result_state.get("citations", [])
            }

            results.append(output_line)

            if args.verbose:
                print("\n--- Output Contract ---")
                print(json.dumps(output_line, indent=2))

            print(f"\n🏁 Question {i} completed")
            print("=" * 70)

        # Save all results to JSONL
        with open(args.out, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

        print(f"\n✅ Saved results to {args.out}")
        print(f"🎉 All {len(questions)} questions processed successfully!")
        print("=" * 70)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)