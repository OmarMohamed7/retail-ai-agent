"""
Main entry point for the application.
"""

import json
import sys

from agent.config import AgentConfig
from agent.rag.retrieval import DocumentRetriever


print("Running agent hybrid...")

if __name__ == "__main__":
    try:
        # Load configuration from environment
        config = AgentConfig()

        # Initialize document retriever
        document_retriever = DocumentRetriever(config)

        # Retrieve chunks for query
        query = "What is the return window for unopened Beverages?"
        raw_chunks = document_retriever.retrieve(query)

        serialized_chunks = []
        for chunk in raw_chunks:
            name = chunk.get("name") or chunk.get("source") or "unknown"
            score = chunk.get("score")
            content = chunk.get("content") or ""
            serialized_chunks.append(
                {
                    "id": chunk.get("id"),
                    "name": name,
                    "score": score,
                    "content": content,
                }
            )

        response = {
            "query": query,
            "chunks": serialized_chunks,
            "total_chunks": len(serialized_chunks),
        }
        print(json.dumps(response, indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
