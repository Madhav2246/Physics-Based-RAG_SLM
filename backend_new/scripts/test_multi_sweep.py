"""
test_multi_sweep.py
-------------------
Tests multi-variable / multi-curve plotting:
  1. Family of W/L vs Vov curves at 3 different Id values  (param family)
  2. gm vs Vov for 3 different W/L ratios                  (param family)
  3. Vth vs Vsb for 3 body-effect coefficients (gamma)     (param family)
  4. Two targets on one plot: Id and gm vs Vov             (dual-Y axes)

Run from backend_new/:
  python scripts/test_multi_sweep.py
"""
import sys, io, copy
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from physics.sweep_engine import (
    parse_sweep_request, parse_multi_curve_param,
    SweepEngine, _UNIT_SCALE, _VAR_UNIT,
)
from physics.exploration_engine import ExplorationEngine
from physics.equation_validator import EquationValidator
from physics.value_tracker import ValueTracker, TrackedValue

validator = EquationValidator()
engine    = ExplorationEngine(validator.symbols)
sweep_eng = SweepEngine()
OUT_DIR   = ROOT / "data" / "evaluation" / "plots_gen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEP = "-" * 72
passed = failed = 0

def _fail(label, reason):
    global failed
    print(f"       FAIL  {label}: {reason}")
    failed += 1

def _pass(label):
    global passed
    print(f"       PASS  {label}")
    passed += 1

# ── helper: run one sweep with a specific fixed param override ────────────────
def _run_one(lhs, rhs, query, req, override_param=None, override_val=None):
    """
    Runs solve_sweep. Optionally injects override_param=override_val into
    the tracker after extraction so we can vary a fixed param across curves.
    """
    import copy as _copy
    result = engine.solve_sweep(lhs, rhs, query, req)
    if result.get("error") or result.get("symbolic_expr") is None:
        return None, result.get("error", "solve_sweep failed")

    sym_expr = result["symbolic_expr"]

    # Build tracker; inject override before run_sweep
    tracker = engine.extract_user_values(query)
    if override_param and override_val is not None:
        tracker.add_user(override_param, override_val, "", f"family param override")

    sr = sweep_eng.run_sweep(req, sym_expr, tracker, result.get("target_var") or req.target_var)
    if sr.error:
        return None, sr.error
    return sr, None


print(f"\n{'='*72}")
print(f"  MULTI-CURVE SWEEP TEST")
print(f"{'='*72}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 1: W/L vs Vov — family over Id = 0.5mA, 1mA, 2mA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("  [T1] W/L vs Vov — family over Id = 0.5mA, 1mA, 2mA")
QUERY1   = "Plot W/L versus Vov from 0.1V to 0.6V for Id=0.5mA, 1mA, 2mA"
CORPUS1  = "Id = 0.5 * mu * Cox * WL * Vov^2"

multi = parse_multi_curve_param(QUERY1)
req1  = parse_sweep_request(QUERY1)

if multi is None:
    _fail("T1", "parse_multi_curve_param returned None")
elif req1 is None:
    _fail("T1", "parse_sweep_request returned None")
else:
    param, values = multi
    print(f"       multi-param: {param} = {values}")
    lhs1, rhs1, msg = validator.validate(CORPUS1)
    if lhs1 is None:
        _fail("T1", f"corpus parse: {msg}")
    else:
        results, labels = [], []
        ok = True
        for v in values:
            label_val = f"{v*1e3:.1f} mA"
            sr, err = _run_one(lhs1, rhs1, QUERY1, req1,
                               override_param=param, override_val=v)
            if err:
                _fail(f"T1 Id={label_val}", err); ok = False; break
            results.append(sr)
            labels.append(f"Id = {label_val}")

        if ok:
            out = str(OUT_DIR / "multi_WL_vs_Vov_Id_family.png")
            sweep_eng.plot_multi(results, labels, out,
                                 title="W/L vs Vov — Id family (0.5/1/2 mA)")
            kb = Path(out).stat().st_size // 1024
            print(f"       curves: {len(results)}  |  saved {kb}KB -> {Path(out).name}")
            _pass("T1")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 2: gm vs Vov — family over W/L = 5, 15, 30
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("  [T2] gm vs Vov — family over W/L = 5, 15, 30")
QUERY2  = "Plot gm vs Vov from 0.1V to 0.8V for WL=5, 15, 30"
CORPUS2 = "gm = mu * Cox * WL * Vov"

multi2 = parse_multi_curve_param(QUERY2)
req2   = parse_sweep_request(QUERY2)

if multi2 is None:
    _fail("T2", "parse_multi_curve_param returned None")
elif req2 is None:
    _fail("T2", "parse_sweep_request returned None")
else:
    param2, values2 = multi2
    print(f"       multi-param: {param2} = {values2}")
    lhs2, rhs2, msg = validator.validate(CORPUS2)
    if lhs2 is None:
        _fail("T2", f"corpus parse: {msg}")
    else:
        results2, labels2 = [], []
        ok2 = True
        for v in values2:
            sr, err = _run_one(lhs2, rhs2, QUERY2, req2,
                               override_param=param2, override_val=v)
            if err:
                _fail(f"T2 WL={v}", err); ok2 = False; break
            results2.append(sr)
            labels2.append(f"W/L = {v:.0f}")

        if ok2:
            out2 = str(OUT_DIR / "multi_gm_vs_Vov_WL_family.png")
            sweep_eng.plot_multi(results2, labels2, out2,
                                 title="gm vs Vov — W/L family (5 / 15 / 30)")
            kb = Path(out2).stat().st_size // 1024
            print(f"       curves: {len(results2)}  |  saved {kb}KB -> {Path(out2).name}")
            _pass("T2")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 3: Vth vs Vsb — family over gamma = 0.2, 0.4, 0.6
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("  [T3] Vth vs Vsb — family over gamma = 0.2, 0.4, 0.6")
QUERY3  = "Plot Vth versus Vsb from 0V to 3V for gamma=0.2, 0.4, 0.6"
CORPUS3 = "Vth = Vth0 + gamma * (sqrt(Phi_f + Vsb) - sqrt(Phi_f))"

multi3 = parse_multi_curve_param(QUERY3)
req3   = parse_sweep_request(QUERY3)

if multi3 is None:
    _fail("T3", "parse_multi_curve_param returned None")
elif req3 is None:
    _fail("T3", "parse_sweep_request returned None")
else:
    param3, values3 = multi3
    print(f"       multi-param: {param3} = {values3}")
    lhs3, rhs3, msg = validator.validate(CORPUS3)
    if lhs3 is None:
        _fail("T3", f"corpus parse: {msg}")
    else:
        results3, labels3 = [], []
        ok3 = True
        for v in values3:
            sr, err = _run_one(lhs3, rhs3, QUERY3, req3,
                               override_param=param3, override_val=v)
            if err:
                _fail(f"T3 gamma={v}", err); ok3 = False; break
            results3.append(sr)
            labels3.append(f"gamma = {v} V^0.5")

        if ok3:
            out3 = str(OUT_DIR / "multi_Vth_vs_Vsb_gamma_family.png")
            sweep_eng.plot_multi(results3, labels3, out3,
                                 title="Vth vs Vsb — body-effect gamma family (0.2 / 0.4 / 0.6)")
            kb = Path(out3).stat().st_size // 1024
            print(f"       curves: {len(results3)}  |  saved {kb}KB -> {Path(out3).name}")
            _pass("T3")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 4: dual-Y — Id and gm vs Vov on same figure, two Y axes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("  [T4] Dual-Y axes — Id AND gm vs Vov on one plot")
QUERY4A  = "Plot Id versus Vov from 0.1V to 0.7V for W/L=20"
CORPUS4A = "Id = 0.5 * mu * Cox * WL * Vov^2"
QUERY4B  = "Plot gm versus Vov from 0.1V to 0.7V for W/L=20"
CORPUS4B = "gm = mu * Cox * WL * Vov"

req4a = parse_sweep_request(QUERY4A)
req4b = parse_sweep_request(QUERY4B)
lhsA, rhsA, msgA = validator.validate(CORPUS4A)
lhsB, rhsB, msgB = validator.validate(CORPUS4B)

if req4a is None or req4b is None:
    _fail("T4", "parse_sweep_request returned None")
elif lhsA is None or lhsB is None:
    _fail("T4", f"corpus parse: {msgA or msgB}")
else:
    srA, errA = _run_one(lhsA, rhsA, QUERY4A, req4a, "WL", 20.0)
    srB, errB = _run_one(lhsB, rhsB, QUERY4B, req4b, "WL", 20.0)
    if errA or errB:
        _fail("T4", errA or errB)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        color_id = "#E74C3C"
        color_gm = "#2980B9"

        ax1.plot(srA.x, srA.y, color=color_id, linewidth=2.2, label="Id")
        ax1.set_xlabel(f"Vov  [V]", fontsize=11)
        ax1.set_ylabel(f"Id  [A]", color=color_id, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_id)

        ax2 = ax1.twinx()
        ax2.plot(srB.x, srB.y, color=color_gm, linewidth=2.2, linestyle="--", label="gm")
        ax2.set_ylabel(f"gm  [A/V]", color=color_gm, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_gm)

        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=10, loc="upper left")
        ax1.set_title("Id  and  gm  vs  Vov  (W/L = 20, dual Y-axis)", fontsize=12)
        ax1.grid(True, alpha=0.2)

        out4 = str(OUT_DIR / "dual_y_Id_gm_vs_Vov.png")
        fig.tight_layout()
        fig.savefig(out4, dpi=130)
        plt.close(fig)
        kb = Path(out4).stat().st_size // 1024
        print(f"       dual-Y saved {kb}KB -> {Path(out4).name}")
        _pass("T4")
print()

# ── Summary ───────────────────────────────────────────────────────────────────
print(SEP)
print(f"  Result: {passed}/{passed+failed} passed, {failed} failed")
print(SEP)
print(f"\n  Plots saved to: {OUT_DIR}")
for p in sorted(OUT_DIR.glob("multi_*.png")) + sorted(OUT_DIR.glob("dual_*.png")):
    kb = p.stat().st_size // 1024
    print(f"    {p.name}  ({kb}KB)")
print()
