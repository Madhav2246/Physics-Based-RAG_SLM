from __future__ import annotations
import json
import datetime
import os
import utils.config as cfg


class RAGLogger:
    """
    Append-only JSONL audit logger for every RAG query/response cycle.

    Fixes applied:
    - Logs ALL result fields: numerical_validation, confidence_score,
      confidence_label, uncertainty_score, stability_label (previously missing).
    - Timestamp now uses UTC ISO-8601 format (datetime.utcnow().isoformat() + 'Z').
    - os.makedirs uses the log file's parent directory (not a hardcoded 'logs/').
    """

    def __init__(self, log_file: str = None):
        self.log_file = log_file or cfg.LOG_FILE
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(
        self,
        query: str,
        evidence: list,
        response: str,
        symbolic_val: str,
        dimension_val: str,
        numerical_val: str = None,
        confidence_score: float = None,
        confidence_label: str = None,
        uncertainty_score: float = None,
        stability_label: str = None,
        semantic_similarity: float = None,
    ) -> None:

        entry = {
            "timestamp":            datetime.datetime.utcnow().isoformat() + "Z",
            "query":                query,
            "retrieved_evidence":   evidence,
            "model_response":       response,
            "symbolic_validation":  symbolic_val,
            "dimension_validation": dimension_val,
            "numerical_validation": numerical_val,
            "semantic_similarity":  round(semantic_similarity, 4) if semantic_similarity is not None else None,
            "confidence_score":     confidence_score,
            "confidence_label":     confidence_label,
            "uncertainty_score":    uncertainty_score,
            "stability_label":      stability_label,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")