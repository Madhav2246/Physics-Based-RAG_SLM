import utils.config as cfg
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import CrossEncoderReranker


class HybridRetriever:
    """
    Dense (FAISS) + Sparse (BM25) retrieval with cross-encoder reranking.

    Fixes applied:
    - set() union replaced with dict.fromkeys() — preserves insertion order
      and is deterministic across runs (set() ordering is hash-dependent).
    - top_k driven by cfg.TOP_K; caller can override per-query.
    """

    def __init__(self):
        self.dense   = DenseRetriever()
        self.sparse  = SparseRetriever()
        self.reranker = CrossEncoderReranker()

    def build_index(self, documents: list[str]) -> None:
        self.dense.build_index(documents)
        self.sparse.build_index(documents)

    def retrieve(self, query: str, top_k: int = None) -> list[str]:
        top_k = top_k if top_k is not None else cfg.TOP_K

        # Retrieve more candidates for better fusion overlap
        dense_results  = self.dense.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # k=60 is standard for RRF
        k = 60
        for rank, doc in enumerate(dense_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank)
            
        for rank, doc in enumerate(sparse_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + 1.0 / (k + rank)

        # Sort by RRF score and take top candidates for reranker
        combined = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        combined = combined[:top_k * 2]

        # Rerank using cross-encoder
        return self.reranker.rerank(query, combined, top_k)