import statistics

class UncertaintyEngine:

    def evaluate(self, responses):

        # Simple diversity metric: unique response ratio
        unique_count = len(set(responses))
        total = len(responses)

        diversity = unique_count / total

        if diversity == 1:
            return 1.0, "HIGH STABILITY"

        if diversity >= 0.6:
            return 0.7, "MODERATE STABILITY"

        return 0.4, "LOW STABILITY"
