import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics.equation_validator import EquationValidator
v = EquationValidator()

good = []
with open('data/evaluation/nvidia_golden_qa.jsonl', encoding='utf-8') as f:
    for i, line in enumerate(f):
        item = json.loads(line)
        eq = item.get('equation', '')
        if not eq:
            continue
        lhs, rhs, msg = v.validate(eq)
        if lhs is not None and '[OK]' in msg:
            good.append((i, item))
        if len(good) >= 8:
            break

print(f'Found {len(good)} questions with parseable corpus equations:\n')
for rank, (idx, item) in enumerate(good):
    diff = item.get('difficulty', '?')
    q = item['question'][:65]
    eq = item.get('equation', '')[:65]
    print(f'  [{rank+1}] Line {idx+1} [{diff}]: {q}')
    print(f'        Corpus eq : {eq}')
    print()
