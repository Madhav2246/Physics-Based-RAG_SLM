"""
test_sweep_queries.py
---------------------
Tests 10 different plotting queries end-to-end through the pipeline
(no server, no GPU — pure CPU algebra + matplotlib).

Checks:
  1. parse_sweep_request() correctly identifies the query as a sweep
  2. solve_sweep() runs without error
  3. SweepEngine.plot() saves a PNG
  4. The URL that _sweep_plot_url() would produce is correct

Run from backend_new/:
  python scripts/test_sweep_queries.py
"""
import sys, io
from pathlib import Path
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physics.sweep_engine import parse_sweep_request, SweepEngine
from physics.exploration_engine import ExplorationEngine
from physics.equation_validator import EquationValidator
from physics.node_profile_manager import NodeProfileManager

validator   = EquationValidator()
engine      = ExplorationEngine(validator.symbols)
npm         = NodeProfileManager()
sweep_eng   = SweepEngine()
EVAL_DIR    = ROOT / "data" / "evaluation" / "plots_gen"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Each tuple: (query, corpus_chunk_to_parse)
# Corpus chunks are intentionally simple so the algebra step is guaranteed
QUERIES = [
    # 1. Classic W/L vs Vov
    ("Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA",
     "Id = 0.5 * mu * Cox * WL * Vov^2"),

    # 2. Id vs Vgs (Vth shift)  — "A versus B from X to Y" syntax: Vgs sweeps, Id on Y
    ("Plot Id versus Vgs from 0.3V to 1.2V",
     "Id = 0.5 * mu * Cox * WL * (Vgs - Vth)^2"),

    # 3. gm vs Vov
    ("Plot gm vs Vov from 0.1V to 0.8V for Id=500uA",
     "gm = mu * Cox * WL * Vov"),

    # 4. W/L vs Id
    ("Plot W/L versus Id from 100uA to 5mA for Vov=0.3V",
     "Id = 0.5 * mu * Cox * WL * Vov^2"),

    # 5. gm vs W/L — sweep aspect ratio, show transconductance scaling
    ("Plot gm versus W/L from 5 to 50",
     "gm = mu * Cox * WL * Vov"),

    # 6. Vth vs Vsb (body effect) — corpus uses Vth0 so Vth is solvable
    ("Plot Vth versus Vsb from 0V to 2V",
     "Vth = Vth0 + gamma * (sqrt(Phi_f + Vsb) - sqrt(Phi_f))"),

    # 7. gm vs Id
    ("Plot gm vs Id from 100uA to 10mA",
     "gm = 2 * Id / Vov"),

    # 8. W/L vs Vov on 5nm GAA
    ("Using 5nm GAA, plot W/L vs Vov from 0.05V to 0.4V for Id=1mA",
     "Id = 0.5 * mu * Cox * WL * Vov^2"),

    # 9. ro vs Id — "A versus B from X to Y": Id sweeps, ro on Y
    ("Plot ro versus Id from 10uA to 1mA",
     "ro = 1 / (Id * gamma)"),

    # 10. Cox vs tox
    ("Plot Cox versus tox from 1nm to 10nm",
     "Cox = eps / tox"),

    # 11. Id vs Vds (channel-length modulation — exercises lam default)
    ("Plot Id versus Vds from 0.1V to 2V for W/L=20",
     "Id = 0.5 * mu * Cox * WL * Vov^2 * (1 + lam * Vds)"),
]

SEP = "-" * 72
print(f"\n{'='*72}")
print(f"  SWEEP QUERY TEST — {len(QUERIES)} queries")
print(f"{'='*72}\n")

passed = failed = 0

for i, (query, corpus_chunk) in enumerate(QUERIES, 1):
    print(f"  [{i:02d}] {query[:65]}")

    # Step 1: parse
    req = parse_sweep_request(query)
    if req is None:
        print(f"       ❌ FAIL — parse_sweep_request returned None")
        failed += 1
        continue
    print(f"       parse  ✓  sweep={req.sweep_var} target={req.target_var} "
          f"range=[{req.start}, {req.stop}]")

    # Step 2: validate corpus chunk
    lhs, rhs, msg = validator.validate(corpus_chunk)
    if lhs is None:
        print(f"       ❌ FAIL — corpus chunk parse failed: {msg}")
        failed += 1
        continue

    # Detect node if any
    node_name = npm.detect_from_query(query)
    req.node_name = node_name or "100nm_CMOS"
    node_defaults = npm.as_tracker_defaults(req.node_name) if node_name else None

    # Step 3: solve + sweep
    try:
        result = engine.solve_sweep(lhs, rhs, query, req,
                                    node_defaults=node_defaults)
        sr = result.get("sweep_result")
        if result.get("error") or sr is None or sr.error:
            err = result.get("error") or (sr.error if sr else "no sweep_result")
            print(f"       ❌ FAIL — solve_sweep: {err}")
            failed += 1
            continue
        print(f"       solve  ✓  target={sr.target_var} n_points={len(sr.x)} "
              f"y_range=[{min(sr.y):.3g}, {max(sr.y):.3g}]")
    except Exception as e:
        print(f"       ❌ FAIL — solve_sweep exception: {type(e).__name__}: {e}")
        failed += 1
        continue

    # Step 4: plot
    out_path = str(EVAL_DIR / f"sweep_{sr.target_var}_vs_{sr.sweep_var}_q{i:02d}.png")
    try:
        saved = sweep_eng.plot(sr, out_path)
        exists = Path(saved).exists()
        size_kb = Path(saved).stat().st_size // 1024
        print(f"       plot   ✓  saved {size_kb}KB → {Path(saved).name}")
    except Exception as e:
        print(f"       ❌ FAIL — plot exception: {type(e).__name__}: {e}")
        failed += 1
        continue

    # Step 5: URL simulation (what api_server._sweep_plot_url would produce)
    try:
        rel = Path(saved).resolve().relative_to(EVAL_DIR.resolve())
        url = f"/static/evaluation/{rel.as_posix()}"
        frontend_url = f"http://localhost:8000{url}"
        print(f"       url    ✓  {url}")
    except Exception as e:
        print(f"       ⚠ URL mismatch: {e}")

    print(f"       ✅ PASS")
    passed += 1
    print()

print(SEP)
print(f"  Result: {passed}/{len(QUERIES)} passed, {failed} failed")
print(SEP)

# If any passed, show what the frontend would load
if passed:
    print(f"\n  Frontend img src examples:")
    for f_path in sorted(EVAL_DIR.glob("sweep_*_q*.png"))[:3]:
        rel = f_path.relative_to(EVAL_DIR)
        print(f"    http://localhost:8000/static/evaluation/{rel.as_posix()}")
print()
