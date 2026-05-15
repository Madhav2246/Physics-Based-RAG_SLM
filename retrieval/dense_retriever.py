from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

class DenseRetriever:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def build_index(self, documents, save_path="data/embeddings/"):
        self.documents = documents

        embeddings = self.embed_model.encode(documents)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

        os.makedirs(save_path, exist_ok=True)

        faiss.write_index(self.index, save_path + "dense.index")
        with open(save_path + "docs.pkl", "wb") as f:
            pickle.dump(documents, f)

    def load_index(self, load_path="data/embeddings/"):
        self.index = faiss.read_index(load_path + "dense.index")
        with open(load_path + "docs.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, query, top_k=3):
        q_emb = self.embed_model.encode([query])
        D, I = self.index.search(np.array(q_emb), top_k)
        
        results = [self.documents[i] for i in I[0]]
        return list(set(results))  # remove duplicates