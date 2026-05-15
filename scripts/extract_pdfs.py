import os
import fitz  # PyMuPDF
from tqdm import tqdm

RAW_DIR = "data/raw_pdfs"
CORPUS_DIR = "data/corpus"

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)
    
    pdf_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))
                
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        return

    print(f"Found {len(pdf_files)} PDFs. Extracting text...")
    
    for pdf_path in tqdm(pdf_files):
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(CORPUS_DIR, f"{base_name}.txt")
        
        # Extract and save
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)

if __name__ == "__main__":
    main()
