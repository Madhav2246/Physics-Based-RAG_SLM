import pytest
import os
import sympy as sp
from pathlib import Path
from physics.sweep_engine import parse_sweep_request, SweepEngine, SweepRequest
from physics.value_tracker import ValueTracker

CASES = [
    ("Plot W/L versus Vov from 0.1V to 0.6V for Id=1mA", "Vov", 0.1, 0.6),
    ("sweep Id from 1uA to 10mA",                         "Id",  1e-6, 1e-2),
    ("what W/L for Id=1mA, Vov=0.3V",                     None,  None, None),
    ("Vov vs W/L, range 0.2 to 0.8",                      "Vov", 0.2, 0.8),
    ("Plot gm versus Vov from 0.2V to 0.5V",              "Vov", 0.2, 0.5),
    ("sweep tox from 1nm to 5nm",                         "tox", 1e-9, 5e-9),
    ("Define threshold voltage",                          None,  None, None),
]

@pytest.mark.parametrize("query, exp_var, exp_start, exp_stop", CASES)
def test_parse_sweep_request(query, exp_var, exp_start, exp_stop):
    req = parse_sweep_request(query)
    if exp_var is None:
        assert req is None
    else:
        assert req is not None
        assert req.sweep_var == exp_var
        assert req.start == pytest.approx(exp_start, rel=1e-9)
        assert req.stop == pytest.approx(exp_stop, rel=1e-9)

def test_run_sweep_and_plot(tmp_path):
    # Dummy setup for run_sweep
    x, y, z = sp.symbols('x y z')
    expr = 2 * x + y  # Target expression
    tracker = ValueTracker()
    tracker.add_user("y", 5.0, "V")
    
    req = SweepRequest(
        sweep_var="x",
        target_var="z",
        start=0.0,
        stop=10.0,
        n_points=10,
        node_name="100nm_CMOS"
    )
    
    engine = SweepEngine()
    result = engine.run_sweep(req, expr, tracker, "z")
    
    # Check output shape (SweepResult uses .x and .y)
    assert len(result.x) == 10
    assert len(result.y) == 10
    assert result.x[0] == pytest.approx(0.0)
    assert result.x[-1] == pytest.approx(10.0)
    assert result.y[0] == pytest.approx(5.0)   # 2*0 + 5
    assert result.y[-1] == pytest.approx(25.0)  # 2*10 + 5
    assert result.error == ""
    
    # Check plot creation
    plot_path = tmp_path / "test_plot.png"
    out_path = engine.plot(result, str(plot_path), title="Test Plot")
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0
