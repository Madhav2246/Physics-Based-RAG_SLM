import json
import datetime
import os

class RAGLogger:

    def __init__(self, log_file="logs/rag_logs.jsonl"):
        self.log_file = log_file
        os.makedirs("logs", exist_ok=True)

    def log(self, query, evidence, response, symbolic_val, dimension_val):

        log_entry = {
            "timestamp": str(datetime.datetime.now()),
            "query": query,
            "retrieved_evidence": evidence,
            "model_response": response,
            "symbolic_validation": symbolic_val,
            "dimension_validation": dimension_val
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")