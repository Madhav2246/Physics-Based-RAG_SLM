class UncertaintyEngine:
    """
    Measures model output stability across multiple sampled responses.

    Fix applied — logic was inverted:
    - OLD: diversity=1.0 (all unique) → "HIGH STABILITY"  ← WRONG
    - NEW: agreement=1.0 (all identical) → "HIGH STABILITY" ← CORRECT

    The agreement score goes from 1.0 (all responses identical, stable model)
    to 0.0 (all responses different, unstable/uncertain model).

    Responses are normalized (stripped + lowercased) before comparison so that
    responses identical in content but differing in trailing whitespace are not
    incorrectly counted as different.
    """

    def evaluate(self, responses: list[str]) -> tuple[float, str]:
        if not responses:
            return 0.0, "LOW STABILITY"

        # Normalize to avoid false uniqueness from whitespace differences
        normalized = [r.strip().lower() for r in responses]
        total = len(normalized)
        unique_count = len(set(normalized))

        # agreement = 1.0 when all same; 0.0 when all different
        # Formula: 1 - (unique - 1) / (total - 1)
        if total == 1:
            agreement = 1.0
        else:
            agreement = 1.0 - (unique_count - 1) / (total - 1)

        agreement = round(max(0.0, min(1.0, agreement)), 3)

        if agreement >= 0.8:
            return agreement, "HIGH STABILITY"
        elif agreement >= 0.5:
            return agreement, "MODERATE STABILITY"
        return agreement, "LOW STABILITY"
