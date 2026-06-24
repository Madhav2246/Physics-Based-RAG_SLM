import json
from pathlib import Path
import shutil

EVAL_DIR = Path("f:/AMRITA ALL SEMESTER/SEMESTER-6/NLP/Physics_Based_RAG_SLM/Physics_Based_RAG_SLM/backend_new/data/evaluation")
OUT_DIR = Path("f:/AMRITA ALL SEMESTER/SEMESTER-6/NLP/Physics_Based_RAG_SLM/Physics_Based_RAG_SLM/evaluation_diff_token_size")

# Delete old files
if OUT_DIR.exists():
    for f in OUT_DIR.glob("*"):
        if f.is_file():
            f.unlink()
else:
    OUT_DIR.mkdir(parents=True)

budgets = [128, 256, 384, 512]
metrics = {}

for b in budgets:
    p = EVAL_DIR / f"stage1_sys_{b}T.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        
        # Calculate truncation and avg_words from per_question
        truncs = 0
        words = 0
        n = len(data["per_question"])
        for q in data["per_question"]:
            ans = q["answer"]
            words += len(ans.split())
            if ans.strip() and ans.strip()[-1] in ("-", "+", "*", "/", "=", ",", "(", "[", "{", "\\"):
                truncs += 1
            else:
                last = ans.split()[-1] if ans.split() else ""
                if last and last[-1].isalnum() and not ans.strip().endswith((".", "?", "!")):
                    truncs += 1
                    
        m = data["summary"]
        m["truncation_rate"] = truncs / n if n > 0 else 0
        m["avg_words"] = words / n if n > 0 else 0
        metrics[b] = m

lines = [
    "# Token Budget Sensitivity Study (Kaggle P100)",
    "",
    "## Results",
    "",
    "| Token Limit | Avg Score | Parse% | Dim% | Num% | Cov% | Trunc% | Avg Words |",
    "|-------------|-----------|--------|------|------|------|--------|-----------|"
]

for b in budgets:
    if b in metrics:
        m = metrics[b]
        lines.append(f"| {b}T | {m['avg_score']:.3f} | {m['parseable']}% | {m['dimensional']}% | {m['numerical']}% | {m['coverage']}% | {m['truncation_rate']*100:.1f}% | {m['avg_words']:.1f} |")

report_path = OUT_DIR / "evaluation_report_diff_tokensize.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Report written to {report_path}")
