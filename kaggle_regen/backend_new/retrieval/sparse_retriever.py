import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi


BM25_DOCS_JSON = Path("data/embeddings/bm25_docs.json")


class SparseRetriever:
    """
    BM25 keyword retrieval.

    Fix applied:
    - Replaced doc.split() (whitespace-only) with a physics-aware regex tokenizer.
      Previously "Vgs-Vth" was one token; now it correctly produces ["Vgs", "Vth"].
      Numbers with scientific notation (e.g. 1e-7) are also preserved as tokens.
    - build_index() prefers bm25_docs.json (written by IngestionEngine) over
      re-tokenising raw docs, so ingested PDFs are visible automatically.
    """

    _TOKEN_RE = re.compile(
        r'[A-Za-z][A-Za-z0-9_]*'           # identifiers: Vgs, Cox, MOSFET…
        r'|[\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?'  # numbers: 1e-7, 0.5, 300
    )

    def __init__(self):
        self.documents: list[str] = []
        self._tokenised: list[list[str]] = []
        self.bm25 = None

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN_RE.findall(text)

    def build_index(self, documents: list[str]) -> None:
        disk_tokenised = self._load_tokenised_from_disk()
        if disk_tokenised:
            self._tokenised = disk_tokenised
            self.documents = list(documents)  # display corpus fallback
        else:
            self.documents = list(documents)
            self._tokenised = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self._tokenised)

    def build_index_from_docs(self, documents: list[str]) -> None:
        """Build from an explicit raw-string list (used when DenseRetriever
        has already loaded the full doc list from disk)."""
        self.documents = list(documents)
        self._tokenised = [self._tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self._tokenised)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.documents[i] for i in ranked if i < len(self.documents)]

    def _load_tokenised_from_disk(self) -> list[list[str]]:
        if BM25_DOCS_JSON.exists():
            return json.loads(BM25_DOCS_JSON.read_text(encoding="utf-8"))
        return []