"""
Check what the RAG pipeline actually retrieves for physics questions.
Run this BEFORE any prompt changes to know if equations are in the chunks.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()
retriever.dense.load_index()
retriever.sparse.build_index_from_docs(retriever.dense.documents)

# Test 3 different question types
test_questions = [
    "What is the equation for MOSFET drain current in saturation?",
    "What is the equation for the current density in a metal-semiconductor contact?",
    "What is the threshold voltage body effect equation?",
]

for q in test_questions:
    print("=" * 70)
    print(f"Q: {q}")
    chunks = retriever.retrieve(q, top_k=3)
    for i, chunk in enumerate(chunks):
        print(f"\n  [Chunk {i+1}] (first 300 chars):")
        print(f"  {chunk[:300]}")
        # Check if there's an equation-like pattern in the chunk
        has_eq = "=" in chunk and any(c.isalpha() for c in chunk)
        print(f"  Has equation pattern: {has_eq}")
    print()
