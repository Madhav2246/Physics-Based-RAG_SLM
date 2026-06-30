import zipfile
import os
from pathlib import Path

# The directories and files we absolutely need to run stage1_physics.py on Kaggle
REQUIRED_PATHS = [
    "backend_new/physics",
    "backend_new/pipeline",
    "backend_new/reasoning",
    "backend_new/retrieval",
    "backend_new/utils",
    "backend_new/scripts/stage1_physics.py",
    "backend_new/requirements.txt",
    "backend_new/models/finetuned_slm",
    "backend_new/data/embeddings",
    "backend_new/data/evaluation/nvidia_golden_qa.jsonl"
]

def create_zip():
    zip_path = "kaggle_stage1_package.zip"
    print(f"Creating {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in REQUIRED_PATHS:
            p = Path(item)
            if p.is_file():
                print(f"Adding file: {item}")
                zipf.write(item)
            elif p.is_dir():
                print(f"Adding directory: {item}")
                for root, dirs, files in os.walk(item):
                    # Skip __pycache__
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Skip large unused files if any sneak in, but embeddings/lora are needed
                        zipf.write(file_path)
            else:
                print(f"WARNING: Could not find {item}")
                
    print(f"\nDone! Upload '{zip_path}' to Kaggle.")

if __name__ == "__main__":
    create_zip()
