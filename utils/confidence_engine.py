class ConfidenceEngine:

    def score(self, evidence,
              symbolic_validation,
              dimension_validation,
              response,
              numerical_validation):

        score = 0.0

        # Retrieval quality
        if len(evidence) >= 2:
            score += 0.2

        # Symbolic parsing
        if "✔" in symbolic_validation:
            score += 0.2

        # Dimensional correctness
        if "✔" in dimension_validation:
            score += 0.25

        # Numerical realism
        if "✔" in numerical_validation:
            score += 0.25

        # Response sanity length
        if 30 < len(response) < 800:
            score += 0.1

        return round(score, 2)

    def interpret(self, score):

        if score >= 0.85:
            return "HIGH CONFIDENCE"
        elif score >= 0.6:
            return "MODERATE CONFIDENCE"
        else:
            return "LOW CONFIDENCE — REVIEW REQUIRED"