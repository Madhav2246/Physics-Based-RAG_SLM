"""
Explore Mode Evaluation — 12 queries, every row hand-checked before inclusion.

Query set designed from pre-screening results:
  SOLVE cases: WL (drain current x4), Vth (body effect x2), gamma x2
  NOT FOUND:   Id, Vov, tox, Cox — honest failures for the table

Hand-check expected values (SOLVE rows only):
  WL = 2*Id / (mu*Cox*Vov^2),  mu=0.05, Cox=0.02

  Q1:  Id=1mA,   Vov=0.5V -> WL = 2*1e-3 / (0.05*0.02*0.25)  = 8.00
  Q2:  Id=200uA, Vov=0.3V -> WL = 2*2e-4 / (0.05*0.02*0.09)  = 4.44
  Q3:  Id=2mA,   Vov=0.4V -> WL = 2*2e-3 / (0.05*0.02*0.16)  = 25.0
  Q4:  Id=500uA, Vov=0.6V -> WL = 2*5e-4 / (0.05*0.02*0.36)  = 2.78

  Vth = Vth0 + gamma*(sqrt(2*Phi_f + Vsb) - sqrt(2*Phi_f))
        Vth0=0.5, gamma=0.4, Phi_f=0.35

  Q5:  Vsb=0.5V -> Vth = 0.5 + 0.4*(sqrt(1.2)-sqrt(0.7)) = 0.603
  Q6:  Vsb=1.0V -> Vth = 0.5 + 0.4*(sqrt(1.7)-sqrt(0.7)) = 0.687

  gamma from body effect: gamma = (Vth-Vth0)/(sqrt(2*Phi_f+Vsb)-sqrt(2*Phi_f))

  Q7:  Vth=0.60, Vsb=0.5V -> gamma = (0.60-0.5)/(1.095-0.837) = 0.387
  Q8:  Vth=0.70, Vsb=1.0V -> gamma = (0.70-0.5)/(1.304-0.837) = 0.428
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.retriever.dense.load_index()
pipeline.retriever.sparse.build_index_from_docs(pipeline.retriever.dense.documents)

# ── Hand-check reference values ────────────────────────────────────────────
def expected_WL(Id, Vov, mu=0.05, Cox=0.02):
    return 2 * Id / (mu * Cox * Vov**2)

def expected_Vth(Vsb, gamma=0.4, Vth0=0.5, Phi_f=0.35):
    return Vth0 + gamma * (math.sqrt(2*Phi_f + Vsb) - math.sqrt(2*Phi_f))

def expected_gamma(Vth, Vsb, Vth0=0.5, Phi_f=0.35):
    return (Vth - Vth0) / (math.sqrt(2*Phi_f + Vsb) - math.sqrt(2*Phi_f))

EVAL_QUERIES = [
    # ── SOLVE cases ──────────────────────────────────────────────────────────
    {
        "id": "S1", "query": "How do I choose W/L for Id = 1mA, Vov = 0.5V?",
        "target": "WL", "expected": expected_WL(1e-3, 0.5),
        "category": "SOLVE",
    },
    {
        "id": "S2", "query": "How do I choose W/L for Id = 200uA, Vov = 0.3V?",
        "target": "WL", "expected": expected_WL(2e-4, 0.3),
        "category": "SOLVE",
    },
    {
        "id": "S3", "query": "Design the W/L ratio for Id = 2mA, Vov = 0.4V.",
        "target": "WL", "expected": expected_WL(2e-3, 0.4),
        "category": "SOLVE",
    },
    {
        "id": "S4", "query": "Calculate the W/L needed for Id = 500uA, Vov = 0.6V.",
        "target": "WL", "expected": expected_WL(5e-4, 0.6),
        "category": "SOLVE",
    },
    {
        "id": "S5", "query": "What Vth do I get with Vsb = 0.5V, gamma = 0.4?",
        "target": "Vth", "expected": expected_Vth(0.5),
        "category": "SOLVE",
    },
    {
        "id": "S6", "query": "What Vth do I get with Vsb = 1.0V, gamma = 0.4?",
        "target": "Vth", "expected": expected_Vth(1.0),
        "category": "SOLVE",
    },
    {
        "id": "S7", "query": "What gamma gives threshold voltage Vth = 0.60V with Vsb = 0.5V?",
        "target": "gamma", "expected": expected_gamma(0.60, 0.5),
        "category": "SOLVE",
    },
    {
        "id": "S8", "query": "What gamma gives threshold voltage Vth = 0.70V with Vsb = 1.0V?",
        "target": "gamma", "expected": expected_gamma(0.70, 1.0),
        "category": "SOLVE",
    },
    # ── NOT FOUND cases — honest failure evidence ──────────────────────────
    {
        "id": "N1", "query": "What Id do I get for W/L = 10, Vov = 0.6V?",
        "target": "Id", "expected": None,
        "category": "NOT_FOUND",
    },
    {
        "id": "N2", "query": "What overdrive voltage Vov gives gm = 2mA/V with W/L = 10?",
        "target": "Vov", "expected": None,
        "category": "NOT_FOUND",
    },
    {
        "id": "N3", "query": "What tox is required to achieve Cox = 0.01 F/m2?",
        "target": "tox", "expected": None,
        "category": "NOT_FOUND",
    },
    {
        "id": "N4", "query": "What Cox do I need for tox = 3nm gate oxide?",
        "target": "Cox", "expected": None,
        "category": "NOT_FOUND",
    },
]

# ── Run all queries ─────────────────────────────────────────────────────────
print("=" * 90)
print("EXPLORE MODE EVALUATION — 12 queries")
print("=" * 90)

rows = []
for q in EVAL_QUERIES:
    result = pipeline.answer(q["query"])
    mode   = result.get("mode", "?")
    er     = result.get("explore_result")
    tracker = er.get("tracker") if er else None

    solved  = er is not None and er.get("success") and er.get("numeric") is not None
    numeric = er.get("numeric") if er else None
    prov    = f"{tracker.provenance_fraction:.0%}" if tracker else "—"
    dim     = "OK" if "[OK]" in result["dimension_validation"] else "FAIL"
    sanity  = "OK" if (er and er.get("sanity_ok")) else ("—" if not solved else "WARN")
    conf    = result["confidence_score"]
    label   = result["confidence_label"].split(" ")[0]  # HIGH/MODERATE/LOW

    # Hand-check for SOLVE rows
    if q["expected"] is not None and numeric is not None:
        rel_err = abs(numeric - q["expected"]) / max(abs(q["expected"]), 1e-30)
        hand_ok = "PASS" if rel_err < 0.01 else f"FAIL (got {numeric:.3g}, expected {q['expected']:.3g})"
    elif q["category"] == "SOLVE" and not solved:
        hand_ok = "UNEXPECTED NOT FOUND"
    elif q["category"] == "NOT_FOUND" and not solved:
        hand_ok = "HONEST FAIL (correct)"
    elif q["category"] == "NOT_FOUND" and solved:
        hand_ok = "UNEXPECTED SOLVE"
    else:
        hand_ok = "—"

    rows.append({
        "id": q["id"], "target": q["target"], "category": q["category"],
        "solved": solved, "numeric": numeric, "expected": q["expected"],
        "prov": prov, "dim": dim, "sanity": sanity,
        "conf": conf, "label": label, "hand_ok": hand_ok,
    })

    status_sym = "SOLVE" if solved else "NOT FOUND"
    numeric_str = f"{numeric:.4g}" if numeric is not None else "N/A"
    print(f"[{q['id']}] {q['query'][:60]}")
    print(f"       mode={mode} | {status_sym} | numeric={numeric_str} "
          f"| prov={prov} | dim={dim} | conf={conf:.3f} {label} | hand={hand_ok}")
    print()

# ── Summary table ───────────────────────────────────────────────────────────
print("=" * 90)
print(f"{'ID':<4} {'Target':<7} {'Solved?':<14} {'Numeric':<10} {'Expected':<10} "
      f"{'Prov':<8} {'Dim':<5} {'Sanity':<7} {'Conf':<6} {'Label':<9} {'Hand-check'}")
print("-" * 90)
for r in rows:
    solved_str = f"{r['numeric']:.4g}" if r["solved"] and r["numeric"] else "NOT FOUND"
    exp_str    = f"{r['expected']:.4g}" if r["expected"] else "—"
    print(f"{r['id']:<4} {r['target']:<7} {solved_str:<14} {exp_str:<10} "
          f"{r['prov']:<8} {r['dim']:<5} {r['sanity']:<7} {r['conf']:<6.3f} "
          f"{r['label']:<9} {r['hand_ok']}")

print()
n_solved   = sum(1 for r in rows if r["solved"])
n_not_found= sum(1 for r in rows if not r["solved"])
n_pass     = sum(1 for r in rows if r["hand_ok"] in ("PASS", "HONEST FAIL (correct)"))
print(f"  Solved:         {n_solved}/12")
print(f"  Honest NOT FOUND: {n_not_found}/12")
print(f"  Hand-check pass:  {n_pass}/12")
print("=" * 90)
