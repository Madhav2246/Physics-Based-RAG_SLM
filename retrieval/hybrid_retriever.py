from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever
from retrieval.reranker import CrossEncoderReranker

class HybridRetriever:

    def __init__(self):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.reranker = CrossEncoderReranker()

    def build_index(self, documents):
        self.dense.build_index(documents)
        self.sparse.build_index(documents)

    def retrieve(self, query, top_k=3):

        dense_results = self.dense.retrieve(query, top_k)
        sparse_results = self.sparse.retrieve(query, top_k)

        combined = list(set(dense_results + sparse_results))

        # 🔥 Rerank using cross-encoder
        reranked = self.reranker.rerank(query, combined, top_k)

        return reranked