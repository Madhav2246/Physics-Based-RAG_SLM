from __future__ import annotations
import numpy as np
from pathlib import Path
import utils.config as cfg

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from retrieval.hybrid_retriever import HybridRetriever
from reasoning.slm_model import TinySLM
from reasoning.prompt_builder import build_prompt
from physics.equation_validator import EquationValidator
from physics.dimension_checker import DimensionChecker
from physics.numerical_validator import NumericalValidator
from physics.exploration_engine import detect_mode, ExplorationEngine
from physics.node_profile_manager import NodeProfileManager
from utils.logger import RAGLogger
from utils.confidence_engine import ConfidenceEngine
from utils.uncertainty_engine import UncertaintyEngine


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class RAGPipeline:
    """
    10-step RAG orchestrator for physics Q&A with deterministic validation.

    Changes from original:
    - All hardcoded constants replaced by cfg.* values.
    - Step 7: computes semantic similarity (answer vs evidence) via the
      dense retriever's embed_model — passed to confidence engine to detect
      factually wrong answers that pass structural checks.
    - Logger now receives all result fields (was only receiving 5 of 11).
    - retrieve() call passes cfg.TOP_K explicitly.
    """

    def __init__(self):
        # Retrieval
        self.retriever = HybridRetriever()

        # Language model
        self.slm = TinySLM()

        # Physics validators
        self.validator          = EquationValidator()
        self.dimension_checker  = DimensionChecker()
        self.numerical_validator = NumericalValidator()

        # Utilities
        self.logger             = RAGLogger()
        self.confidence_engine  = ConfidenceEngine()
        self.uncertainty_engine = UncertaintyEngine()
        self.exploration_engine = ExplorationEngine(self.validator.symbols)
        self.node_manager       = NodeProfileManager()
        # Wire the already-loaded SLM into the two-stage value extractor.
        # No new weights are loaded — self.slm is reused.
        self.exploration_engine.set_slm_model(self.slm)

    def build(self, documents: list[str]) -> None:
        """Build retrieval indexes from a list of document strings."""
        self.retriever.build_index(documents)

    def _find_corpus_equation(self, evidence: list):
        """
        Scan retrieved chunks for the first parseable, physics-relevant equation.
        Three quality filters reject OCR garbage / PDF artifacts.
        Returns (lhs_expr, rhs_expr, equation_str) or (None, None, None).
        Filter 3 (MOSFET-only symbols) intentionally removed — it blocked
        extraction for all non-MOSFET domains (Fowler-Nordheim, BJT, etc.).
        """
        import sympy as _sp
        for chunk in evidence:
            c_lhs, c_rhs, c_msg = self.validator.validate(chunk)
            if c_lhs is None or "[OK]" not in c_msg:
                continue
            if not isinstance(c_lhs, _sp.Basic) or not isinstance(c_rhs, _sp.Basic):
                continue                                   # Filter 0: real exprs
            if c_lhs.is_number and c_rhs.is_number:
                continue                                   # Filter 1: not 2=3
            all_syms = c_lhs.free_symbols | c_rhs.free_symbols
            if len(all_syms) < 2:
                continue                                   # Filter 2: >=2 symbols
            return c_lhs, c_rhs, f"{c_lhs} = {c_rhs}"
        return None, None, None

    @staticmethod
    def _clean_design_query(query: str) -> str:
        """
        Strip sweep/plot/node boilerplate and numeric ranges from a design query
        so retrieval embeds on the physics terms, not the directive noise.
        Used only as a retrieval-retry fallback when the verbose query misses.
        """
        import re
        q = re.sub(r'using\s+\d+nm[\w\s.-]*?node|using\s+\d+nm[\w\s-]*', ' ', query, flags=re.I)
        q = re.sub(r'\bplot\b|\bsweep\b|\bversus\b|\bvs\.?\b', ' ', q, flags=re.I)
        q = re.sub(r'from\s+[\d.]+\s*\w*\s+to\s+[\d.]+\s*\w*', ' ', q, flags=re.I)
        q = re.sub(r'range\s+[\d.]+\s+to\s+[\d.]+', ' ', q, flags=re.I)
        q = re.sub(r'\s+', ' ', q).strip(' ,.')
        return q

    def answer(self, query: str) -> dict:
        import re as _re
        def _strip_latex(text: str) -> str:
            if not text:
                return ""
            # Replace LaTeX math wrappers with spaces, preserving the content inside
            text = text.replace(r'\[', ' ').replace(r'\]', ' ')
            text = text.replace(r'\(', ' ').replace(r'\)', ' ')
            text = text.replace('$$', ' ').replace('$', ' ')
            # Translate common LaTeX ops to Python
            text = text.replace('\\frac', '/').replace('\\sqrt', 'sqrt')
            text = text.replace('\\cdot', '*').replace('\\times', '*')
            # Remove remaining backslash commands (e.g. \text, \mu, etc.)
            text = _re.sub(r'\\[a-zA-Z]+', ' ', text)
            return text

        # -- 1️⃣  Retrieve Evidence --------------------------------------------
        evidence = self.retriever.retrieve(query, top_k=cfg.TOP_K)

        # -- 2  Extract equation directly from corpus chunks (deterministic) --
        corpus_lhs_expr, corpus_rhs_expr, corpus_equation = \
            self._find_corpus_equation(evidence)

        # -- 3  Mode detection: SWEEP vs EXPLORE vs LOOKUP ───────────────────
        # SWEEP:   query asks to plot a variable over a range (most specific).
        # EXPLORE: query asks to solve/design for a single point.
        # LOOKUP:  query asks what an equation is (default, safe path).
        # The LOOKUP path is completely unchanged — SWEEP and EXPLORE are additive.
        mode = detect_mode(query)
        explore_result = None
        sweep_plot_path = None

        # Retrieval retry: verbose design queries ("Using 5nm GAA, plot W/L vs
        # Vov from 0.1 to 0.6 for Id=1mA") embed poorly and miss the governing
        # equation. If a design-mode query found no corpus equation, re-retrieve
        # on a cleaned query stripped of plot/sweep/range/node boilerplate.
        # LOOKUP queries never trigger this — their retrieval path is unchanged.
        if corpus_lhs_expr is None and mode in ("SWEEP", "EXPLORE"):
            cleaned = self._clean_design_query(query)
            if cleaned and cleaned.lower() != query.lower():
                retry_ev = self.retriever.retrieve(cleaned, top_k=cfg.TOP_K)
                r_lhs, r_rhs, r_eq = self._find_corpus_equation(retry_ev)
                if r_lhs is not None:
                    corpus_lhs_expr, corpus_rhs_expr, corpus_equation = r_lhs, r_rhs, r_eq
                    evidence = retry_ev   # use the equation-bearing evidence

        # Detect process-node profile (e.g. "5nm GAA") before solving so the
        # exploration engine resolves unknowns from node-specific constants.
        # None => generic 100nm baseline (unchanged behavior).
        node_name = self.node_manager.detect_from_query(query)
        node_defaults = (self.node_manager.as_tracker_defaults(node_name)
                         if node_name else None)

        # Honesty guard: a design query with no governing equation cannot be
        # solved or swept. Relabel to LOOKUP so the returned `mode` reflects
        # what actually happens (SLM answers from evidence), not a SWEEP/EXPLORE
        # that silently produced nothing.
        if mode in ("SWEEP", "EXPLORE") and corpus_lhs_expr is None:
            mode = "LOOKUP"

        # ── SWEEP mode ────────────────────────────────────────────────────────
        if mode == "SWEEP" and corpus_lhs_expr is not None:
            from physics.sweep_engine import parse_sweep_request
            sweep_req = parse_sweep_request(query)
            if sweep_req is not None:
                sweep_req.node_name = node_name or "100nm_CMOS"
                try:
                    explore_result = self.exploration_engine.solve_sweep(
                        corpus_lhs_expr, corpus_rhs_expr, query,
                        sweep_req, node_defaults=node_defaults,
                    )
                    sr = explore_result.get("sweep_result")
                    if explore_result.get("error") or sr is None or sr.error:
                        mode = "EXPLORE"          # fall back to single point
                        explore_result = None
                    else:
                        out = (_PROJECT_ROOT / "data" / "evaluation" /
                               f"sweep_{sweep_req.target_var or 'result'}_vs_"
                               f"{sweep_req.sweep_var}.png")
                        sweep_plot_path = self.exploration_engine._sweep_engine.plot(
                            sr, str(out)
                        )
                        explore_result["sweep_plot_path"] = sweep_plot_path
                except Exception as exc:
                    print(f"[pipeline] SWEEP failed ({exc}) — falling back to EXPLORE.")
                    mode = "EXPLORE"
                    explore_result = None
            else:
                mode = "EXPLORE"

        # ── EXPLORE mode (also the SWEEP fallback target) ─────────────────────
        if mode == "EXPLORE" and corpus_lhs_expr is not None and explore_result is None:
            explore_result = self.exploration_engine.solve(
                corpus_lhs_expr, corpus_rhs_expr, query, node_defaults=node_defaults
            )
            if explore_result.get("error"):
                # Solve failed or target not found — fall back silently to LOOKUP
                mode = "LOOKUP"
                explore_result = None

        # -- 4  Compose response ─────────────────────────────────────────────
        if mode in ("EXPLORE", "SWEEP") and explore_result is not None:
            # Deterministic algebra path — response from format_response() or
            # solve_sweep(). Must start with "Equation: ..." for validator scan.
            response = explore_result.get('response', '')
            if not response.startswith('Equation:'):
                # Fallback: prepend derived form in validator-compatible format
                derived = explore_result.get('symbolic', 'NOT FOUND IN CORPUS')
                response = f"Equation: {derived}\n" + response
            responses = [response]
        else:
            # === LOOKUP mode: Best-of-N Neuro-Symbolic Re-ranking ===
            prompt = build_prompt(query, evidence, corpus_equation=corpus_equation)
            responses = self.slm.generate_multiple(prompt, n_samples=cfg.N_SAMPLES)
            
            # Evaluate all candidates and select the best one based on physics score and semantic similarity tie-breaker
            best_candidate = responses[0]
            best_score = -1
            best_sim = -1.0
            
            evidence_str = " ".join(evidence)
            evid_emb = None
            try:
                embed = self.retriever.dense.embed_model
                evid_emb = embed.encode([evidence_str], show_progress_bar=False)[0]
            except Exception:
                pass
            
            for resp in responses:
                clean_resp = _strip_latex(resp)
                # 1. Symbolic validation
                lhs, rhs, sym_msg = self.validator.validate(clean_resp)
                sym_ok = (lhs is not None and "[OK]" in sym_msg)
                
                # 2. Dimensional validation
                dim_ok = False
                if lhs is not None:
                    dim_msg = self.dimension_checker.check_equation(lhs, rhs)
                    dim_ok = "[OK]" in dim_msg
                
                # 3. Numerical validation & symbol coverage
                num_ok = False
                cov_ok = False
                if rhs is not None:
                    num_msg = self.numerical_validator.evaluate(lhs, rhs)
                    num_ok = "[OK]" in num_msg
                    cov_ok = num_ok and "Unresolved" not in num_msg
                
                candidate_physics_score = (1 if sym_ok else 0) + (1 if dim_ok else 0) + (1 if num_ok else 0) + (1 if cov_ok else 0)
                
                # Calculate semantic similarity
                sim = 0.0
                if evid_emb is not None:
                    try:
                        resp_emb = embed.encode([resp], show_progress_bar=False)[0]
                        sim = _cosine_similarity(resp_emb, evid_emb)
                    except Exception:
                        pass
                
                # Choose the candidate with the highest validation score, break ties with semantic similarity
                if candidate_physics_score > best_score:
                    best_score = candidate_physics_score
                    best_sim = sim
                    best_candidate = resp
                elif candidate_physics_score == best_score:
                    if sim > best_sim:
                        best_sim = sim
                        best_candidate = resp
            
            # Promote best candidate to model_response and responses[0]
            model_response = best_candidate
            if model_response in responses:
                responses.remove(model_response)
            responses.insert(0, model_response)

            if corpus_equation:
                response = f"Equation: {corpus_equation}\n\n{model_response}"
            else:
                response = f"Equation: NOT FOUND IN CORPUS\n\n{model_response}"
            responses[0] = response

        # -- LaTeX sanitise before validator sees response ─────────────────────
        # The 0.5B model often ignores anti-LaTeX instructions and emits \[...\]
        # or \(...\) blocks that crash SymPy with TypeError on subscripts.
        # Strip/translate those constructs so the equation validator can still
        # find and parse any valid Python-notation equation in the same text.
        response = _strip_latex(response)

        # Validate the full response (which contains either corpus or model generated equation)
        validation_target = response

        # -- 4️⃣  Symbolic Validation ------------------------------------------
        lhs_expr, rhs_expr, symbolic_validation = self.validator.validate(validation_target)

        # -- 5️⃣  Dimensional Validation ---------------------------------------
        if lhs_expr is not None:
            dimension_validation = self.dimension_checker.check_equation(
                lhs_expr, rhs_expr
            )
        else:
            dimension_validation = "[WARN] Dimension check skipped."

        # -- 6️⃣  Numerical Sanity Check ---------------------------------------
        if rhs_expr is not None:
            numerical_validation = self.numerical_validator.evaluate(lhs_expr, rhs_expr)
        else:
            numerical_validation = "[WARN] Numerical check skipped."

        # -- 7️⃣  Semantic Similarity (answer ↔ evidence) ----------------------
        # Detects factually wrong answers whose structure passes validation.
        semantic_similarity = None
        try:
            embed = self.retriever.dense.embed_model
            resp_emb  = embed.encode([response],          show_progress_bar=False)
            evid_emb  = embed.encode([" ".join(evidence)], show_progress_bar=False)
            semantic_similarity = _cosine_similarity(resp_emb[0], evid_emb[0])
        except Exception:
            pass  # non-fatal — confidence engine handles None gracefully

        # -- 8  Confidence Score + Breakdown ------------------------------------
        if mode in ("EXPLORE", "SWEEP") and explore_result and explore_result.get('success'):
            # Explore/Sweep mode uses its own scorer — semantic similarity excluded
            # because derived equations don't match corpus chunks by design.
            tracker = explore_result.get('tracker')
            confidence_score, confidence_breakdown = self.confidence_engine.score_explore(
                corpus_found      = corpus_lhs_expr is not None,
                unique_solution   = bool(explore_result.get('symbolic')),
                dim_ok            = "[OK]" in dimension_validation,
                sanity_ok         = bool(explore_result.get('sanity_ok')),
                symbolic_ok       = "[OK]" in symbolic_validation,
                provenance_fraction = tracker.provenance_fraction if tracker else 0.0,
            )
        else:
            # LOOKUP mode: standard compute() with semantic similarity
            confidence_score, confidence_breakdown = self.confidence_engine.compute(
                evidence,
                symbolic_validation,
                dimension_validation,
                response,
                numerical_validation,
                semantic_similarity=semantic_similarity,
            )
        confidence_label = self.confidence_engine.interpret(confidence_score)

        # -- 9️⃣  Uncertainty / Stability Estimation ---------------------------
        uncertainty_score, stability_label = self.uncertainty_engine.evaluate(responses)

        # -- 🔟  Logging -------------------------------------------------------
        self.logger.log(
            query, evidence, response,
            symbolic_validation, dimension_validation,
            numerical_val=numerical_validation,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            uncertainty_score=uncertainty_score,
            stability_label=stability_label,
            semantic_similarity=semantic_similarity,
        )

        # -- Return structured result dict -------------------------------------
        return {
            "response":             response,
            "all_responses":        responses,
            "evidence":             evidence,
            "symbolic_validation":  symbolic_validation,
            "dimension_validation": dimension_validation,
            "numerical_validation": numerical_validation,
            "semantic_similarity":  semantic_similarity,
            "confidence_score":     confidence_score,
            "confidence_label":     confidence_label,
            "confidence_breakdown": confidence_breakdown,
            "uncertainty_score":    uncertainty_score,
            "stability_label":      stability_label,
            "mode":                 mode,
            "explore_result":       explore_result,      # None in LOOKUP mode
            "sweep_plot_path":      sweep_plot_path,     # None unless mode == SWEEP
            "node_profile":         node_name or "100nm_CMOS (default)",
        }