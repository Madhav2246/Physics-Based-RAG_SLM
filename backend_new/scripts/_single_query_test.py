"""
Test the copy-from-context prompt on MOSFET drain current question
where the chunk is known to contain the clean equation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.retriever.dense.load_index()
pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)

question = "What is the equation for MOSFET drain current in saturation?"
print(f"Q: {question}\n")

result = pipeline.answer(question)

print("Evidence retrieved:")
for i, chunk in enumerate(result["evidence"]):
    print(f"  [Chunk {i+1}]: {chunk[:120]}")

print(f"\nRAG response:\n  {result['response']}")
print(f"\nSymbolic  : {result['symbolic_validation']}")
print(f"Dimension : {result['dimension_validation']}")
print(f"Numerical : {result['numerical_validation']}")
print(f"Confidence: {result['confidence_score']} -> {result['confidence_label']}")
