import json
import os

embeddings_dir = r"d:\S6\NLP\Physics_Based_RAG_SLM\backend_new\data\embeddings"
docs_path = os.path.join(embeddings_dir, "docs.json")

if os.path.exists(docs_path):
    with open(docs_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    total_chunks = len(docs)
    total_words = sum(len(d.split()) for d in docs)
    total_chars = sum(len(d) for d in docs)

    # Count actual extracted equations (simple regex for lines with equals signs or latex delimiters)
    eq_count = 0
    for d in docs:
        # Check how many distinct equations appear
        # We can scan for = signs that look like math
        lines = d.split('\n')
        for line in lines:
            if '=' in line and any(sym in line for sym in ['+', '-', '*', '/', '^', '_', 'V', 'I', 'q', 'k', 'T']):
                eq_count += 1

    print(f"SUCCESS")
    print(f"Chunks: {total_chunks}")
    print(f"Words: {total_words}")
    print(f"Chars: {total_chars}")
    print(f"Equations: {eq_count}")
else:
    print(f"File not found: {docs_path}")
