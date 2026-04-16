"""
Layer 5 — Retrieval-Augmented Generation (RAG)
Three components from the diagram:
  - Vector Store  (FAISS index)
  - Retriever     (top-k nearest neighbour search)
  - Context Injector (formats retrieved docs for prompt injection)
"""

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import numpy as np


class RAGModule:
    """
    Stores reasoning traces, task docs, and successful plans in a FAISS
    vector store, then retrieves the most relevant entries to ground each
    reasoning step.

    query = goal + current_reasoning   (as shown in the diagram)
    """

    def __init__(self, embedder=None, dim: int = 384):
        """
        Initialize RAG module.
        
        Args:
            embedder: Optional embedding function (if None, uses heuristic retrieval)
            dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)
        """
        self.embedder  = embedder
        self.dim = dim
        
        if FAISS_AVAILABLE and embedder:
            self.index = faiss.IndexFlatL2(dim)
        else:
            self.index = None
            
        self.documents = []   # raw text entries

    # ------------------------------------------------------------------ #
    #  Vector Store — add documents                                        #
    # ------------------------------------------------------------------ #

    def add_document(self, doc: dict) -> None:
        """
        Add a single document to the RAG store.
        
        Args:
            doc: Dictionary with 'id', 'title', 'content', 'source' keys
        """
        if isinstance(doc, dict):
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
        else:
            text = str(doc)
        
        self.documents.append({"text": text, "metadata": doc if isinstance(doc, dict) else {}})
        
        # Add to FAISS index if available
        if self.index and self.embedder:
            try:
                vec = self.embedder.encode(text, convert_to_tensor=False)
                self.index.add(np.array([vec], dtype="float32"))
            except Exception as e:
                print(f"Warning: Could not add document to index: {e}")

    def add_documents(self, docs: list) -> None:
        """Add a batch of documents to the vector store."""
        for doc in docs:
            self.add_document(doc)

    def add_reasoning_trace(self, trace: str) -> None:
        """Store a completed reasoning trace for future retrieval."""
        self.add_document({
            "title": "[TRACE]",
            "content": trace
        })

    def add_successful_plan(self, plan: str) -> None:
        """Store a plan that successfully completed a task."""
        self.add_document({
            "title": "[PLAN]",
            "content": plan
        })

    # ------------------------------------------------------------------ #
    #  Retriever                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, goal: str, current_reasoning: str = "", k: int = 3) -> list:
        """
        Retrieve top-k relevant documents.
        query = goal + current_reasoning  (matches diagram)
        
        Args:
            goal: Goal text to retrieve for
            current_reasoning: Current reasoning step
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if len(self.documents) == 0:
            return []

        # Use FAISS if available, otherwise use keyword matching
        if self.index and self.embedder:
            return self._retrieve_faiss(goal, current_reasoning, k)
        else:
            return self._retrieve_heuristic(goal, current_reasoning, k)

    def _retrieve_faiss(self, goal: str, current_reasoning: str, k: int) -> list:
        """Retrieve using FAISS vector similarity."""
        try:
            query = f"{goal} {current_reasoning}".strip()
            query_vec = self.embedder.encode(query, convert_to_tensor=False)
            query_vec = np.array([query_vec], dtype="float32")

            k_actual = min(k, len(self.documents))
            _, indices = self.index.search(query_vec, k_actual)

            return [self.documents[i]["text"] for i in indices[0] if i < len(self.documents)]
        except Exception as e:
            print(f"Warning: FAISS retrieval failed: {e}")
            return self._retrieve_heuristic(goal, current_reasoning, k)

    def _retrieve_heuristic(self, goal: str, current_reasoning: str, k: int) -> list:
        """Retrieve using keyword matching heuristic."""
        query = f"{goal} {current_reasoning}".lower()
        query_words = set(query.split())

        # Score documents by keyword overlap
        scores = []
        for doc in self.documents:
            doc_text = doc["text"].lower()
            doc_words = set(doc_text.split())
            overlap = len(query_words & doc_words)
            scores.append((overlap, doc["text"]))

        # Sort by score and return top-k
        scores.sort(reverse=True, key=lambda x: x[0])
        return [text for _, text in scores[:k]]

    # ------------------------------------------------------------------ #
    #  Context Injector                                                    #
    # ------------------------------------------------------------------ #

    def inject_context(self, retrieved: list) -> str:
        """
        Formats retrieved documents for injection into the reasoning prompt.
        Maps to the 'Context Injector' component in the diagram.
        
        Args:
            retrieved: List of retrieved documents
            
        Returns:
            Formatted string for prompt injection
        """
        if not retrieved:
            return "No relevant context retrieved."
        lines = [f"  [{i+1}] {doc[:100]}..." if len(doc) > 100 else f"  [{i+1}] {doc}" 
                 for i, doc in enumerate(retrieved)]
        return "Retrieved context:\n" + "\n".join(lines)
