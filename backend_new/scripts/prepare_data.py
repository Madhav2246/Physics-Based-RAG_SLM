import os
import json
from tqdm import tqdm

CORPUS_DIR    = os.path.join("data", "corpus")
PROCESSED_DIR = os.path.join("data", "processed")
CHUNK_SIZE    = 256   # words per chunk (was 512 — truncated to 128 tokens anyway)
OVERLAP       = 32    # overlap words between consecutive chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window word-level chunker with overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.split()) > overlap:   # discard tiny trailing chunks
            chunks.append(chunk)
    return chunks


def format_as_chat(chunk: str) -> str:
    """
    Wrap a physics corpus chunk in the Qwen 2.5 chat template format.

    Previously: {"text": raw_chunk}  — raw pretraining, model doesn't learn Q&A format
    Now: {"text": <Qwen chat template with system/user/assistant turns>}

    The chunk is used as both the "topic" in the user turn and the "answer" in the
    assistant turn, teaching the model to produce physics content in the expected
    instruction-following format.
    """
    return (
        "<|im_start|>system\n"
        "You are a semiconductor device physics assistant. "
        "Answer questions using precise equations and physical reasoning.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Explain the following semiconductor physics concept:\n\n{chunk}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{chunk}\n"
        "<|im_end|>"
    )


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_file = os.path.join(PROCESSED_DIR, "train_dataset.jsonl")

    txt_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"No text files found in {CORPUS_DIR}")
        return

    print(f"Processing {len(txt_files)} files into chat-format chunks...")

    total_chunks = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        for filename in tqdm(txt_files):
            file_path = os.path.join(CORPUS_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            for chunk in chunk_text(text, CHUNK_SIZE, OVERLAP):
                record = {"text": format_as_chat(chunk)}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Created {total_chunks} chat-format chunks → {out_file}")


if __name__ == "__main__":
    main()
