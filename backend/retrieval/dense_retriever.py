from __future__ import annotations
import os
import json
import pickle
from typing import Optional, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import utils.config as cfg


class DenseRetriever:
    """
    Semantic similarity retrieval using FAISS + SentenceTransformers.

    Fixes applied:
    - list(set(results)) → ordered list from FAISS ranking (no set() conversion).
      FAISS returns results in L2-distance order; set() was destroying that ranking.
    - Document storage switched from pickle → JSON (safer, human-readable).
      Backward-compatible: still reads old .pkl files if .json doesn't exist.
    - embeddings cast to float32 explicitly (FAISS requirement).
    - save_path driven by cfg.EMBEDDINGS_PATH.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embed_model = SentenceTransformer(model_name)
        self.index: Optional[Any] = None   # faiss.Index (no public type)
        self.documents: list[str] = []

    def build_index(self, documents: list[str],
                    save_path: str = None) -> None:
        save_path = save_path or cfg.EMBEDDINGS_PATH

        # Prefer disk docs so previously ingested PDFs are included
        disk_docs = self._load_docs_from_disk(save_path)
        self.documents = disk_docs if disk_docs else list(documents)

        index_path = os.path.join(save_path, "dense.index")
        if os.path.exists(index_path):
            loaded_index = faiss.read_index(index_path)
            if loaded_index.ntotal == len(self.documents):
                self.index = loaded_index
                return
            print(f"[DenseRetriever] Index size mismatch "
                  f"({loaded_index.ntotal} vectors vs {len(self.documents)} docs) — rebuilding.")

        embeddings = self.embed_model.encode(
            self.documents, show_progress_bar=False
        )
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype=np.float32))

        os.makedirs(save_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_path, "dense.index"))

        # JSON preferred over pickle — safer and portable
        with open(os.path.join(save_path, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load_index(self, load_path: str = None) -> None:
        load_path = load_path or cfg.EMBEDDINGS_PATH
        self.index = faiss.read_index(os.path.join(load_path, "dense.index"))

        json_path = os.path.join(load_path, "docs.json")
        pkl_path  = os.path.join(load_path, "docs.pkl")

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        elif os.path.exists(pkl_path):    # backward compatibility
            with open(pkl_path, "rb") as f:
                self.documents = pickle.load(f)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        q_emb = self.embed_model.encode([query])
        D, I = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        # Preserve FAISS distance-ranked order — no set() conversion
        return [self.documents[i] for i in I[0] if 0 <= i < len(self.documents)]

    def _load_docs_from_disk(self, load_path: str = None) -> list[str]:
        load_path = load_path or cfg.EMBEDDINGS_PATH
        json_path = os.path.join(load_path, "docs.json")
        pkl_path  = os.path.join(load_path, "docs.pkl")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        return []