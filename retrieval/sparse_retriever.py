from rank_bm25 import BM25Okapi

class SparseRetriever:

    def __init__(self):
        self.documents = []
        self.bm25 = None

    def build_index(self, documents):
        self.documents = documents
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def retrieve(self, query, top_k=3):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [self.documents[i] for i in ranked_indices]