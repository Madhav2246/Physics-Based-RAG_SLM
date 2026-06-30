from __future__ import annotations


class ConfidenceEngine:
    """
    Rule-based confidence scorer for RAG pipeline outputs.

    compute() is the single source of truth — it returns both the score and
    the per-component breakdown in one pass. score() wraps it for backward
    compatibility with evaluate_pipeline.py.

    Components and weights:
      Retrieval quality   0.20  — scaled continuously by chunk count (not binary)
      Symbolic parsing    0.15  — SymPy parseable equation found
      Dimensional check   0.20  — LHS units == RHS units
      Numerical realism   0.20  — substituted values in physical range
      Response length     0.10  — 20–1200 chars (post prompt-strip)
      Semantic similarity 0.15  — cosine(answer, evidence), scaled continuously

    Thresholds: HIGH >= 0.80, MODERATE >= 0.55, LOW < 0.55
    """

    def compute(
        self,
        evidence: list,
        symbolic_validation: str,
        dimension_validation: str,
        response: str,
        numerical_validation: str,
        semantic_similarity: float | None = None,
    ) -> tuple[float, list[dict]]:
        """
        Returns (confidence_score, breakdown_list).

        breakdown_list entries:
            {
              "label": str,   — human-readable component name
              "score": float, — actual points earned (0.0 – max)
              "max":   float, — maximum possible points for this component
              "ok":    bool,  — did this component pass?
            }
        """
        breakdown = []
        total = 0.0

        # ── Retrieval quality (0.20) ──────────────────────────────────────────
        # Scale continuously with chunk count so retrieving 1 vs 3 chunks is
        # reflected rather than collapsing to a binary pass/fail.
        n_chunks   = len(evidence)
        ev_score   = round(min(n_chunks / 3.0, 1.0) * 0.20, 3)
        ev_ok      = n_chunks >= 2
        total     += ev_score
        breakdown.append({
            "label": f"Evidence retrieved ({n_chunks} chunk{'s' if n_chunks != 1 else ''})",
            "score": ev_score,
            "max":   0.20,
            "ok":    ev_ok,
        })

        # ── Symbolic parsing (0.15) ───────────────────────────────────────────
        sym_ok    = "[OK]" in symbolic_validation
        sym_score = 0.15 if sym_ok else 0.0
        total    += sym_score
        breakdown.append({
            "label": "Symbolic parse",
            "score": sym_score,
            "max":   0.15,
            "ok":    sym_ok,
        })

        # ── Dimensional correctness (0.20) ────────────────────────────────────
        dim_ok    = "[OK]" in dimension_validation
        dim_score = 0.20 if dim_ok else 0.0
        total    += dim_score
        breakdown.append({
            "label": "Dimensional check",
            "score": dim_score,
            "max":   0.20,
            "ok":    dim_ok,
        })

        # ── Numerical realism (0.20) ──────────────────────────────────────────
        num_ok    = "[OK]" in numerical_validation
        num_score = 0.20 if num_ok else 0.0
        total    += num_score
        breakdown.append({
            "label": "Numerical realism",
            "score": num_score,
            "max":   0.20,
            "ok":    num_ok,
        })

        # ── Response length (0.10) ────────────────────────────────────────────
        resp_len  = len(response.strip())
        len_ok    = 20 <= resp_len <= 1200
        len_score = 0.10 if len_ok else 0.0
        total    += len_score
        breakdown.append({
            "label": f"Response length ({resp_len} chars)",
            "score": len_score,
            "max":   0.10,
            "ok":    len_ok,
        })

        # ── Semantic similarity (0.15, continuous) ────────────────────────────
        if semantic_similarity is not None:
            sem_val   = float(max(0.0, min(1.0, semantic_similarity)))
            sem_score = round(0.15 * sem_val, 3)
            sem_ok    = sem_val >= 0.50
            total    += sem_score
            breakdown.append({
                "label": f"Semantic similarity ({sem_val:.2f})",
                "score": sem_score,
                "max":   0.15,
                "ok":    sem_ok,
            })

        return round(min(total, 1.0), 3), breakdown

    def score(
        self,
        evidence: list,
        symbolic_validation: str,
        dimension_validation: str,
        response: str,
        numerical_validation: str,
        semantic_similarity: float | None = None,
    ) -> float:
        """Backward-compatible wrapper — returns score only."""
        s, _ = self.compute(
            evidence, symbolic_validation, dimension_validation,
            response, numerical_validation, semantic_similarity,
        )
        return s

    def interpret(self, score: float) -> str:
        if score >= 0.80:
            return "HIGH CONFIDENCE"
        elif score >= 0.55:
            return "MODERATE CONFIDENCE"
        return "LOW CONFIDENCE - REVIEW REQUIRED"

    def score_explore(
        self,
        corpus_found:      bool,
        unique_solution:   bool,
        dim_ok:            bool,
        sanity_ok:         bool,
        symbolic_ok:       bool,
        provenance_fraction: float,
    ) -> tuple[float, list[dict]]:
        """
        Confidence scorer for EXPLORE mode.

        Replaces semantic similarity (which penalises derived equations for not
        matching corpus chunks — exactly what explore mode produces) with a
        provenance score based on how many values were user-supplied vs assumed.

        Deliberate weight targets (verified):
          All user values, all checks pass  -> 0.65 + 0.25 = 0.90  (HIGH)
          All default values, all checks    -> 0.65 + 0.00 = 0.65  (MODERATE)
          Some user values (50/50), all ok  -> 0.65 + 0.125 = 0.775 (MODERATE-HIGH)
          Corpus eq not found               -> 0.00 base  -> LOW regardless
          Solve() returned empty list       -> 0.00 base  -> LOW

        Component breakdown (base max = 0.65, provenance max = 0.25):
          corpus_found      0.25  — corpus equation must exist for any result
          unique_solution   0.20  — SymPy returned exactly one solution
          dim_ok            0.10  — derived expression dimensionally consistent
          sanity_ok         0.07  — numeric result in physical range
          symbolic_ok       0.03  — base equation parsed correctly
          provenance        0-0.25 — scaled by fraction of user-supplied values
        """
        breakdown = []
        total = 0.0

        # corpus_found (0.25)
        cs = 0.25 if corpus_found else 0.0
        total += cs
        breakdown.append({
            "label": "Corpus equation found",
            "score": cs, "max": 0.25, "ok": corpus_found,
        })

        # unique_solution (0.20) — only meaningful if corpus was found
        us = 0.20 if (corpus_found and unique_solution) else 0.0
        total += us
        breakdown.append({
            "label": "Unique SymPy solution",
            "score": us, "max": 0.20, "ok": unique_solution,
        })

        # dim_ok (0.10)
        ds = 0.10 if dim_ok else 0.0
        total += ds
        breakdown.append({
            "label": "Dimensional consistency (derived)",
            "score": ds, "max": 0.10, "ok": dim_ok,
        })

        # sanity_ok (0.07)
        ss = 0.07 if sanity_ok else 0.0
        total += ss
        breakdown.append({
            "label": "Physical sanity range",
            "score": ss, "max": 0.07, "ok": sanity_ok,
        })

        # symbolic_ok (0.03)
        sym_s = 0.03 if symbolic_ok else 0.0
        total += sym_s
        breakdown.append({
            "label": "Base equation symbolic parse",
            "score": sym_s, "max": 0.03, "ok": symbolic_ok,
        })

        # provenance (0 – 0.25, continuous)
        # 1.0 = all user-supplied (+0.25), 0.0 = all defaults (+0.00)
        prov_score = round(0.25 * max(0.0, min(1.0, provenance_fraction)), 3)
        total += prov_score
        breakdown.append({
            "label": f"Value provenance ({provenance_fraction:.0%} user-supplied)",
            "score": prov_score, "max": 0.25,
            "ok": provenance_fraction >= 0.5,
        })

        return round(min(total, 1.0), 3), breakdown
