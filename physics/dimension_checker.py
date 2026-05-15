import sympy as sp


class DimensionChecker:

    def __init__(self):

        # Base dimensions: I, V, L, T
        self.dim_map = {
            "Id": {"I": 1},
            "mu": {"L": 2, "V": -1, "T": -1},
            "Cox": {"I": 1, "T": 1, "V": -1, "L": -2},
            "W": {"L": 1},
            "L": {"L": 1},
            "Vgs": {"V": 1},
            "Vth": {"V": 1}
        }

    def _add(self, d1, d2):
        result = d1.copy()
        for k, v in d2.items():
            result[k] = result.get(k, 0) + v
        return result

    def _scale(self, dims, factor):
        return {k: v * factor for k, v in dims.items()}

    def _simplify(self, dims):
        return {k: v for k, v in dims.items() if v != 0}

    def evaluate(self, expr):

        # Symbol
        if isinstance(expr, sp.Symbol):
            name = expr.name
            if name in self.dim_map:
                return self.dim_map[name]
            else:
                # Unknown symbol: assume dimensionless
                return {}

        # Multiplication
        if isinstance(expr, sp.Mul):
            dims = {}
            for arg in expr.args:
                dims = self._add(dims, self.evaluate(arg))
            return dims

        # Power
        if isinstance(expr, sp.Pow):
            base = expr.args[0]
            exponent = expr.args[1]

            base_dims = self.evaluate(base)

            try:
                exponent = int(exponent)
            except:
                return {}

            return self._scale(base_dims, exponent)

        # Addition
        if isinstance(expr, sp.Add):
            dims_list = [self.evaluate(arg) for arg in expr.args]
            first = self._simplify(dims_list[0])
            for d in dims_list[1:]:
                if self._simplify(d) != first:
                    return {"DIMENSION_MISMATCH": 1}
            return first

        return {}

    def check_equation(self, lhs_expr, rhs_expr):

        lhs_dims = self._simplify(self.evaluate(lhs_expr))
        rhs_dims = self._simplify(self.evaluate(rhs_expr))

        if lhs_dims == rhs_dims:
            return f"✔ Dimensionally consistent: {lhs_dims}"
        else:
            return (
                "❌ Dimension mismatch:\n"
                f"  LHS {lhs_dims}\n"
                f"  RHS {rhs_dims}"
            )