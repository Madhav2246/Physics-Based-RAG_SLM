import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)


class EquationValidator:

    def __init__(self):

        self.symbols = {
            "Id": sp.symbols("Id"),
            "mu": sp.symbols("mu"),
            "Cox": sp.symbols("Cox"),
            "W": sp.symbols("W"),
            "L": sp.symbols("L"),
            "Vgs": sp.symbols("Vgs"),
            "Vth": sp.symbols("Vth")
        }

        self.transformations = (
            standard_transformations +
            (implicit_multiplication_application,)
        )

    def normalize_equation(self, eq):

        # Replace Greek mu and variations
        eq = eq.replace("μ_n", "mu")
        eq = eq.replace("mu_n", "mu")
        eq = eq.replace("μ", "mu")

        # 🔥 CRITICAL FIX: replace ^ with **
        eq = eq.replace("^", "**")

        # Fix merged tokens
        eq = eq.replace("muCox", "mu*Cox")

        # Fix common aliases
        eq = eq.replace("Lg", "L")
        eq = eq.replace("Leff", "L")

        return eq.strip()

    def extract_equation(self, text):
        lines = text.split("\n")
        for line in lines:
            if "=" in line:
                line = line.strip()
                if ":" in line:
                    parts = line.split(":")
                    for part in reversed(parts):
                        if "=" in part:
                            line = part
                            break
                parts = line.split("=")
                if len(parts) == 2:
                    lhs = parts[0].split()[-1]
                    rhs = parts[1].strip()
                    return f"{lhs} = {rhs}"
                return line.strip()
        return None

    def validate(self, text):

        equation_line = self.extract_equation(text)

        if equation_line is None:
            return None, None, "⚠ No equation detected."

        try:
            equation_line = self.normalize_equation(equation_line)

            lhs, rhs = equation_line.split("=")

            lhs_expr = parse_expr(
                lhs.strip(),
                local_dict=self.symbols,
                transformations=self.transformations
            )

            rhs_expr = parse_expr(
                rhs.strip(),
                local_dict=self.symbols,
                transformations=self.transformations
            )

            return lhs_expr, rhs_expr, "✔ Equation parsed successfully"

        except Exception as e:
            return None, None, f"⚠ Parsing failed: {str(e)}"