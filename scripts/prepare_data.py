import os
import json
from tqdm import tqdm

CORPUS_DIR = "data/corpus"
PROCESSED_DIR = "data/processed"
CHUNK_SIZE = 512
OVERLAP = 50

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > overlap:  # Don't keep tiny trailing chunks
            chunks.append(chunk)
    return chunks

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_file = os.path.join(PROCESSED_DIR, "train_dataset.jsonl")
    
    txt_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"No text files found in {CORPUS_DIR}")
        return
        
    print(f"Processing {len(txt_files)} files into chunks...")
    
    total_chunks = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for filename in tqdm(txt_files):
            file_path = os.path.join(CORPUS_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            for chunk in chunks:
                # Format for language modeling (or causal LM)
                record = {"text": chunk}
                out_f.write(json.dumps(record) + "\n")
                total_chunks += 1
                
    print(f"Created {total_chunks} chunks. Saved to {out_file}")

if __name__ == "__main__":
    main()
