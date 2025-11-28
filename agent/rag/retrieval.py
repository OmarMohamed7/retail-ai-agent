"""Document retriever."""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.config import AgentConfig


class DocumentRetriever:
    """Document retriever."""

    def __init__(self, config: AgentConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.chunks = []
        self.chunk_metadata = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.load_documents()

    def load_documents(self):
        """Load documents"""

        # Check if documents directory exists
        if not self.config.docs_dir.exists():
            self.logger.error(
                "Documents directory %s does not exist", self.config.docs_dir
            )
            raise FileNotFoundError(
                f"Documents directory {self.config.docs_dir} does not exist"
            )

        self.logger.info("Loading documents from %s", self.config.docs_dir)

        for file in self.config.docs_dir.glob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = self.chunk_document(content, file.name)
            self.chunks.extend(c["content"] for c in chunks)
            self.chunk_metadata.extend(chunks)

        if self.chunks:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words="english",
                ngram_range=(1, 1), # use only individual words Example text:
                                    # "Unopened beverages have 14 days return window" 
                                    # will be tokenized as ["Unopened", "beverages", "have", "14", "days", "return", "window"]

            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
            self.logger.info(
                "Indexed %d chunks from %d documents",
                len(self.chunks),
                len(set(c["source"] for c in self.chunk_metadata)),
            )
        else:
            self.logger.warning("No documents found in %s", self.config.docs_dir)

    def chunk_document(self, content: str, doc_name: str) -> List[Dict[str, Any]]:
        """Chunk documents"""
        self.logger.info("Chunking documents")

        chunks = []
        lines = content.split("\n")
        current_chunk = []
        current_words = 0
        chunk_idx = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_words = len(line.split())
            if (line.startswith("#") and current_chunk) or (
                current_words + line_words > self.config.chunk_size
            ):
                if current_chunk:
                    text = "\n".join(current_chunk)
                    chunks.append(
                        {
                            "doc_name": doc_name,
                            "chunk_idx": chunk_idx,
                            "content": text,
                            "id": f"{doc_name}-{chunk_idx}",
                            "source": doc_name,
                        }
                    )
                    chunk_idx += 1
                    current_chunk = [line]
                    current_words = line_words
            else:
                current_chunk.append(line)
                current_words += line_words

        # Last chunk
        if current_chunk:
            text = "\n".join(current_chunk)
            chunks.append(
                {
                    "doc_name": doc_name,
                    "chunk_idx": chunk_idx,
                    "content": text,
                    "id": f"{doc_name}-{chunk_idx}",
                    "source": doc_name,
                }
            )
        return chunks

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks"""
        self.logger.info("Retrieving documents for query: %s", query)
        if not self.chunks:
            self.logger.warning("No chunks loaded")
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        top_k_indices = np.argsort(scores)[-self.config.top_k :][::-1]
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0.01:
                results.append(
                    {
                        "id": self.chunk_metadata[idx]["id"],
                        "name": self.chunk_metadata[idx]["doc_name"],
                        "content": self.chunks[idx],
                        "source": self.chunk_metadata[idx]["source"],
                        "score": scores[idx],
                    }
                )

        self.logger.info(
            "Retrieved %d chunks for query: %s",
            len(results),
            query,
        )
        return results
