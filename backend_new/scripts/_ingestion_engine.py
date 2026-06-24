"""
Core ingestion logic (not a CLI entry point).

Responsibilities:
  1. Extract text from a PDF (via PyMuPDF).
  2. Chunk the text (sliding window, sentence-aware).
  3. Append new chunks to the persisted document store  (data/embeddings/docs.json).
  4. Incrementally add new embeddings to the FAISS index (data/embeddings/dense.index).
  5. Rebuild the BM25 token store from the full doc list (data/embeddings/bm25_docs.json).
  6. Record the ingested PDF in the registry           (data/embeddings/registry.json).
"""

import json
import re
import pickle
import datetime
from pathlib import Path
from typing import List, Dict, Any

# Heavy deps imported lazily so --list / --reset work without the ML stack
_fitz                = None
_np                  = None
_faiss               = None
_SentenceTransformer = None
_BM25Okapi           = None


def _ensure_deps():
    global _fitz, _np, _faiss, _SentenceTransformer, _BM25Okapi
    if _fitz is None:
        try:
            import fitz as _m; _fitz = _m
        except ImportError:
            raise ImportError("PyMuPDF not installed — run: pip install pymupdf")
    if _np is None:
        import numpy as _m; _np = _m
    if _faiss is None:
        try:
            import faiss as _m; _faiss = _m
        except ImportError:
            raise ImportError("faiss-cpu not installed — run: pip install faiss-cpu")
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            _SentenceTransformer = _ST
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed — run: pip install sentence-transformers"
            )
    if _BM25Okapi is None:
        try:
            from rank_bm25 import BM25Okapi as _BM25; _BM25Okapi = _BM25
        except ImportError:
            raise ImportError("rank-bm25 not installed — run: pip install rank-bm25")


PROJECT_ROOT   = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
DENSE_INDEX    = EMBEDDINGS_DIR / "dense.index"
DOCS_PKL       = EMBEDDINGS_DIR / "docs.pkl"
DOCS_JSON      = EMBEDDINGS_DIR / "docs.json"
BM25_DOCS_JSON = EMBEDDINGS_DIR / "bm25_docs.json"
REGISTRY_JSON  = EMBEDDINGS_DIR / "registry.json"

# Must match the active index built by rebuild_index_gpu.py (SOTA: bge-large @ 384/64).
# Kept in sync with utils/config.py.
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_DIM        = 1024

CHUNK_WORDS   = 384
OVERLAP_WORDS = 64


class IngestionEngine:
    """
    Stateless helper — each public method loads state from disk, modifies it,
    and writes it back so retrievers always see a consistent view.
    """

    def ingest_pdfs(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Ingest a list of PDF paths.  Returns one result dict per PDF:
          {"filename": str, "status": "ingested"|"skipped"|"failed",
           "chunk_count": int, "error": str|None}
        """
        _ensure_deps()
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        registry  = self.load_registry()
        all_docs  = self._load_docs()
        embed_mdl = _SentenceTransformer(EMBED_MODEL_NAME)
        index     = self._load_or_create_faiss_index()

        results = []
        for pdf_path in pdf_paths:
            pdf_path = Path(pdf_path)
            filename = pdf_path.name

            print(f"\n  {filename}")

            if self._already_ingested(filename, registry):
                print(f"    Already ingested — skipping")
                results.append({"filename": filename, "status": "skipped",
                                 "chunk_count": 0, "error": None})
                continue

            try:
                text = self._extract_text(pdf_path)
            except Exception as exc:
                print(f"    Extraction failed: {exc}")
                results.append({"filename": filename, "status": "failed",
                                 "chunk_count": 0, "error": str(exc)})
                continue

            if not text.strip():
                msg = "PDF produced no extractable text (scanned image?)"
                print(f"    {msg}")
                results.append({"filename": filename, "status": "failed",
                                 "chunk_count": 0, "error": msg})
                continue

            chunks = self._chunk_text(text)
            if not chunks:
                msg = "Chunking produced 0 usable chunks"
                print(f"    {msg}")
                results.append({"filename": filename, "status": "failed",
                                 "chunk_count": 0, "error": msg})
                continue

            print(f"    Extracted {len(chunks)} chunk(s)  "
                  f"(~{sum(len(c.split()) for c in chunks)} words total)")

            print(f"    Embedding...", end="", flush=True)
            embeddings = embed_mdl.encode(
                chunks, show_progress_bar=False, batch_size=32,
                convert_to_numpy=True,
            ).astype("float32")
            index.add(embeddings)
            print(f" done  (index now has {index.ntotal} vectors)")

            all_docs.extend(chunks)

            registry[filename.lower()] = {
                "original_name": filename,
                "chunk_count":   len(chunks),
                "ingested_at":   datetime.datetime.utcnow().isoformat() + "Z",
                "chunk_start":   index.ntotal - len(chunks),
            }

            results.append({"filename": filename, "status": "ingested",
                             "chunk_count": len(chunks), "error": None})

        if any(r["status"] == "ingested" for r in results):
            self._save_faiss_index(index)
            self._save_docs(all_docs)
            self._rebuild_bm25_store(all_docs)
            self._save_registry(registry)
            print("\n  Indexes persisted to disk.")

        return results

    def load_registry(self) -> Dict[str, Any]:
        if REGISTRY_JSON.exists():
            return json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
        return {}

    def reset(self):
        for path in [DENSE_INDEX, DOCS_PKL, DOCS_JSON, BM25_DOCS_JSON, REGISTRY_JSON]:
            if path.exists():
                path.unlink()

    # -- private helpers -------------------------------------------------------

    def _already_ingested(self, filename: str, registry: Dict) -> bool:
        return filename.lower() in registry

    def _extract_text(self, pdf_path: Path) -> str:
        doc = _fitz.open(str(pdf_path))
        pages_text = []
        for page in doc:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (round(b[1] / 20) * 20, b[0]))
            page_text = "\n".join(
                b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()
            )
            pages_text.append(page_text)
        doc.close()
        return "\n\n".join(pages_text)

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]

        chunks: List[str] = []
        window: List[str] = []
        window_wc = 0

        for sent in sentences:
            sent_wc = len(sent.split())
            window.append(sent)
            window_wc += sent_wc

            if window_wc >= CHUNK_WORDS:
                chunk = " ".join(window)
                if len(chunk.split()) >= 30:
                    chunks.append(chunk)
                while window_wc > OVERLAP_WORDS and window:
                    removed = window.pop(0)
                    window_wc -= len(removed.split())

        if window:
            chunk = " ".join(window)
            if len(chunk.split()) >= 30:
                chunks.append(chunk)

        return chunks

    def _load_or_create_faiss_index(self):
        if DENSE_INDEX.exists():
            idx = _faiss.read_index(str(DENSE_INDEX))
            print(f"    Loaded existing FAISS index ({idx.ntotal} vectors)")
            return idx
        print(f"    Creating new FAISS IndexFlatL2 (dim={EMBED_DIM})")
        return _faiss.IndexFlatL2(EMBED_DIM)

    def _save_faiss_index(self, index):
        _faiss.write_index(index, str(DENSE_INDEX))

    def _load_docs(self) -> List[str]:
        if DOCS_JSON.exists():
            return json.loads(DOCS_JSON.read_text(encoding="utf-8"))
        if DOCS_PKL.exists():
            with open(DOCS_PKL, "rb") as f:
                docs = pickle.load(f)
            self._save_docs(docs)
            return docs
        return []

    def _save_docs(self, docs: List[str]):
        DOCS_JSON.write_text(
            json.dumps(docs, ensure_ascii=False, indent=None),
            encoding="utf-8",
        )
        with open(DOCS_PKL, "wb") as f:
            pickle.dump(docs, f)

    def _rebuild_bm25_store(self, docs: List[str]):
        tokenised = [
            re.findall(r'\w+', doc.lower())
            for doc in docs
        ]
        BM25_DOCS_JSON.write_text(
            json.dumps(tokenised, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_registry(self, registry: Dict):
        REGISTRY_JSON.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
