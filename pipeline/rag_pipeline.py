from retrieval.hybrid_retriever import HybridRetriever
from reasoning.slm_model import TinySLM
from reasoning.prompt_builder import build_prompt
from physics.equation_validator import EquationValidator
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator
from utils.logger import RAGLogger
from utils.confidence_engine import ConfidenceEngine
from utils.uncertainty_engine import UncertaintyEngine


class RAGPipeline:

    def __init__(self):

        # Retrieval
        self.retriever = HybridRetriever()

        # Language Model
        self.slm = TinySLM()

        # Validators
        self.validator = EquationValidator()
        self.dimension_checker = DimensionChecker()
        self.numerical_validator = NumericalValidator()

        # Utilities
        self.logger = RAGLogger()
        self.confidence_engine = ConfidenceEngine()
        self.uncertainty_engine = UncertaintyEngine()

    def build(self, documents):
        self.retriever.build_index(documents)

    def answer(self, query):

        # -----------------------------
        # 1️⃣ Retrieve Evidence
        # -----------------------------
        evidence = self.retriever.retrieve(query)

        # -----------------------------
        # 2️⃣ Build Prompt
        # -----------------------------
        prompt = build_prompt(query, evidence)

        # -----------------------------
        # 3️⃣ Multi-Sample Generation
        # -----------------------------
        responses = self.slm.generate_multiple(prompt, n_samples=3)

        # Use first response for physics validation
        response = responses[0]

        # -----------------------------
        # 4️⃣ Symbolic Validation
        # -----------------------------
        lhs_expr, rhs_expr, symbolic_validation = self.validator.validate(response)

        # -----------------------------
        # 5️⃣ Dimensional Validation
        # -----------------------------
        if lhs_expr is not None:
            dimension_validation = self.dimension_checker.check_equation(
                lhs_expr, rhs_expr
            )
        else:
            dimension_validation = "⚠ Dimension check skipped."

        # -----------------------------
        # 6️⃣ Numerical Sanity Validation
        # -----------------------------
        if rhs_expr is not None:
            numerical_validation = self.numerical_validator.evaluate(rhs_expr)
        else:
            numerical_validation = "⚠ Numerical check skipped."

        # -----------------------------
        # 7️⃣ Confidence Score
        # -----------------------------
        confidence_score = self.confidence_engine.score(
            evidence,
            symbolic_validation,
            dimension_validation,
            response,
            numerical_validation
        )

        confidence_label = self.confidence_engine.interpret(confidence_score)

        # -----------------------------
        # 8️⃣ Uncertainty Estimation
        # -----------------------------
        uncertainty_score, stability_label = self.uncertainty_engine.evaluate(responses)

        # -----------------------------
        # 9️⃣ Logging
        # -----------------------------
        self.logger.log(
            query,
            evidence,
            response,
            symbolic_validation,
            dimension_validation
        )

        # -----------------------------
        # 🔟 Return Structured Output
        # -----------------------------
        return {
            "response": response,
            "all_responses": responses,
            "symbolic_validation": symbolic_validation,
            "dimension_validation": dimension_validation,
            "numerical_validation": numerical_validation,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "uncertainty_score": uncertainty_score,
            "stability_label": stability_label,
            "evidence": evidence
        }