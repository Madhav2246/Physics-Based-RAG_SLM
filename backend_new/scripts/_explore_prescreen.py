"""
Pre-screen: which design targets are actually solvable from corpus equations?

For each candidate target, retrieves corpus chunks and tests whether
solve_for() returns a valid symbolic solution. Only targets that pass
here should appear in the Day 2 eval table.

Also tests gamma (sqrt inversion, may have multiple branches) and
tox (needs Cox = eps_ox/tox in corpus, not just a Cox value).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid_retriever import HybridRetriever
from physics.equation_validator import EquationValidator
from physics.exploration_engine import ExplorationEngine
from physics.value_tracker import ValueTracker
import sympy as sp

# Load retriever
retriever = HybridRetriever()
retriever.dense.load_index()
retriever.sparse.build_index_from_docs(retriever.dense.documents)

validator = EquationValidator()
engine    = ExplorationEngine(validator)

# Candidate queries and their intended targets
CANDIDATES = [
    # (natural language query, intended target, notes)
    ("What W/L do I need for Id = 1mA, Vov = 0.5V?",          "WL",    "drain current"),
    ("What W/L for Id = 200uA, Vov = 0.3V?",                  "WL",    "drain current low Id"),
    ("What Id do I get for W/L = 10, Vov = 0.6V?",            "Id",    "drain current solve Id"),
    ("What Vov gives Id = 500uA with W/L = 8?",               "Vov",   "overdrive voltage"),
    ("What Vth do I get with Vsb = 0.5V?",                    "Vth",   "body effect"),
    ("What Vth with Vsb = 1.0V, gamma = 0.4?",               "Vth",   "body effect explicit gamma"),
    ("What gamma gives Vth shift of 0.2V with Vsb = 0.5V?",  "gamma", "invert body effect"),
    ("What tox for Cox = 0.01 F/m2?",                         "tox",   "oxide thickness"),
    ("What Cox for tox = 3nm?",                               "Cox",   "oxide capacitance"),
    ("What Vov gives gm = 2mA/V with W/L = 10?",             "Vov",   "transconductance"),
    ("What W/L gives gm = 1mA/V, Vov = 0.5V?",              "WL",    "transconductance gm"),
]

print("=" * 80)
print("EXPLORE MODE PRE-SCREENING")
print("Checking which targets are reachable from corpus equations")
print("=" * 80)

reachable = []
not_reachable = []

for query, target, note in CANDIDATES:
    chunks = retriever.retrieve(query, top_k=3)
    known_symbols = set(validator.symbols.values())

    # Find best corpus equation (quality filters)
    corpus_lhs, corpus_rhs, corpus_eq_str = None, None, None
    for chunk in chunks:
        c_lhs, c_rhs, c_msg = validator.validate(chunk)
        if c_lhs is None or "[OK]" not in c_msg:
            continue
        if not isinstance(c_lhs, sp.Basic) or not isinstance(c_rhs, sp.Basic):
            continue
        if c_lhs.is_number and c_rhs.is_number:
            continue
        all_syms = c_lhs.free_symbols | c_rhs.free_symbols
        if len(all_syms) < 2:
            continue
        if not (all_syms & known_symbols):
            continue
        corpus_lhs, corpus_rhs = c_lhs, c_rhs
        corpus_eq_str = f"{c_lhs} = {c_rhs}"
        break

    if corpus_lhs is None:
        status = "NO CORPUS EQ"
        not_reachable.append((query, target, note, status, ""))
        print(f"  [SKIP ] {target:<8} {note:<30} -> no corpus equation found")
        continue

    # Test solve
    tracker = ValueTracker()
    result = engine.solve_for(corpus_lhs, corpus_rhs, target, tracker, corpus_eq_str)

    if result['success'] and result['symbolic']:
        # Check for multiple branches (may indicate gamma issue)
        from sympy import solve as sp_solve, Eq
        import sympy as sp2
        target_sym = validator.symbols.get(target)
        if target_sym:
            try:
                all_solutions = sp_solve(
                    Eq(corpus_lhs, corpus_rhs), target_sym
                )
                n_sol = len(all_solutions)
            except Exception:
                n_sol = 1
        else:
            n_sol = 1

        branch_note = f" [{n_sol} solution branch{'es' if n_sol != 1 else ''}]" if n_sol > 1 else ""
        status = f"REACHABLE{branch_note}"
        reachable.append((query, target, note, corpus_eq_str))
        print(f"  [OK   ] {target:<8} {note:<30} -> {result['symbolic'][:60]}{branch_note}")
    else:
        status = f"NOT SOLVABLE: {result['error'][:60]}"
        not_reachable.append((query, target, note, status, corpus_eq_str or ""))
        print(f"  [WARN ] {target:<8} {note:<30} -> {result['error'][:60]}")

print()
print("=" * 80)
print(f"REACHABLE ({len(reachable)}): safe for eval table")
for q, t, n, eq in reachable:
    print(f"  {t:<8} | {n:<30} | corpus: {eq[:50]}")
print()
print(f"NOT REACHABLE ({len(not_reachable)}): will be honest NOT FOUND or needs different corpus")
for q, t, n, s, eq in not_reachable:
    print(f"  {t:<8} | {n:<30} | {s[:60]}")
print("=" * 80)
