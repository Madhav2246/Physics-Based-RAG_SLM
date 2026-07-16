import sys
from pathlib import Path
import sympy as sp
import math

# Add backend directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We import the base DimensionChecker and NumericalValidator to show how they can be sub-classed or customized.
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator

class ThermodynamicsDimensionChecker(DimensionChecker):
    """
    Customized DimensionChecker for Thermodynamics.
    Base dimensions: 
      - L: Length (m)
      - K: Temperature (K)
      - mol: Amount of substance
      - J: Energy
    """
    def __init__(self):
        super().__init__()
        # Override the symbol mapping for thermodynamics variables
        self.dim_map = {
            "P":   {"J": 1, "L": -3},  # Pressure: Energy/Volume = J/m³
            "V":   {"L": 3},           # Volume: m³
            "n":   {"mol": 1},         # Moles: mol
            "R":   {"J": 1, "K": -1, "mol": -1}, # Gas constant: J/(mol*K)
            "T":   {"K": 1},           # Temperature: K
            
            # Constants/Dimensionless
            "pi":  {},                 # dimensionless
        }

class ThermodynamicsNumericalValidator(NumericalValidator):
    """
    Customized NumericalValidator for Thermodynamics.
    Checks values against standard room temperature/pressure states.
    """
    def __init__(self):
        super().__init__()
        # Standard values for 1 mole of ideal gas at STP:
        # P = 101325 Pa, V = 0.0224 m3 (22.4 L), n = 1 mol, R = 8.314 J/mol*K, T = 273.15 K
        self.test_values = {
            "P":   101325.0,  # Pascals
            "V":   0.022414,  # m³
            "n":   1.0,       # moles
            "R":   8.314,     # J/(mol*K)
            "T":   273.15,    # Kelvin
        }

    def evaluate(self, lhs_expr, rhs_expr) -> str:
        try:
            # Substitute test values
            lhs_val = float(lhs_expr.subs(self.test_values).evalf())
            rhs_val = float(rhs_expr.subs(self.test_values).evalf())
            
            # Check for physical temperature constraints (e.g. T >= 0 K)
            T_val = self.test_values.get("T", 0.0)
            if T_val < 0.0:
                return "[FAIL] Numerical validation failed: Absolute zero violated (T < 0 K)"
            
            # Check if LHS and RHS are equal/close under ideal conditions
            diff = abs(lhs_val - rhs_val)
            rel_diff = diff / max(abs(lhs_val), abs(rhs_val), 1e-9)
            
            if rel_diff > 0.05: # allow 5% margin for numerical/ideal deviations
                return f"[WARN] Numerical discrepancy too large: LHS={lhs_val:.3e}, RHS={rhs_val:.3e} (diff={rel_diff*100:.1f}%)"
            
            return f"[OK] Numerically validated: LHS={lhs_val:.3e} ≈ RHS={rhs_val:.3e}"
        except Exception as e:
            return f"[WARN] Evaluation error: {str(e)}"

# Let's run a quick demo test
if __name__ == "__main__":
    print("=== Neuro-Symbolic Validation Generalization Demo (Thermodynamics) ===")
    
    # Instantiate custom checkers
    dim_checker = ThermodynamicsDimensionChecker()
    num_validator = ThermodynamicsNumericalValidator()
    
    # 1. Correct equation: Ideal Gas Law (P*V = n*R*T)
    # We parse LHS and RHS using SymPy
    lhs = sp.sympify("P * V")
    rhs = sp.sympify("n * R * T")
    
    print("\nTest Case 1: Ideal Gas Law (P * V = n * R * T)")
    dims_lhs = dim_checker.evaluate(lhs)
    dims_rhs = dim_checker.evaluate(rhs)
    print(f"LHS Dimensions: {dim_checker._simplify(dims_lhs)}")
    print(f"RHS Dimensions: {dim_checker._simplify(dims_rhs)}")
    
    # Dimensional check
    if dim_checker._simplify(dims_lhs) == dim_checker._simplify(dims_rhs):
        print("[OK] Dimensional consistency check passed!")
    else:
        print("[FAIL] Dimensional consistency check failed!")
        
    # Numerical check
    num_status = num_validator.evaluate(lhs, rhs)
    print(f"Numerical Check Status: {num_status}")
    
    # 2. Incorrect equation (dimensionally wrong): P * V = n * R * T^2
    print("\nTest Case 2: Dimensionally Wrong Equation (P * V = n * R * T^2)")
    lhs2 = sp.sympify("P * V")
    rhs2 = sp.sympify("n * R * T**2")
    dims_lhs2 = dim_checker.evaluate(lhs2)
    dims_rhs2 = dim_checker.evaluate(rhs2)
    print(f"LHS Dimensions: {dim_checker._simplify(dims_lhs2)}")
    print(f"RHS Dimensions: {dim_checker._simplify(dims_rhs2)}")
    
    if dim_checker._simplify(dims_lhs2) == dim_checker._simplify(dims_rhs2):
         print("[OK] Dimensional consistency check passed!")
    else:
         print("[FAIL] Dimensional consistency check failed (as expected)!")
         
    # 3. Physically invalid temperature (T < 0 K)
    print("\nTest Case 3: Violation of Temperature Constraint (T = -10 K)")
    # Temporarily set temperature to a negative value in test values
    num_validator.test_values["T"] = -10.0
    num_status_temp = num_validator.evaluate(lhs, rhs)
    print(f"Numerical Check Status: {num_status_temp}")
