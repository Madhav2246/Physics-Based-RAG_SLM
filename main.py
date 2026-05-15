import pathlib
_original_read_text = pathlib.Path.read_text
def _utf8_read_text(self, encoding=None, errors=None):
    if encoding is None:
        encoding = 'utf-8'
    return _original_read_text(self, encoding=encoding, errors=errors)
pathlib.Path.read_text = _utf8_read_text

import builtins
_original_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_open(*args, **kwargs)
builtins.open = _utf8_open

import os
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
import utils.config
from pipeline.rag_pipeline import RAGPipeline

import traceback

print("Starting main.py...")

if __name__ == "__main__":
    try:
        pipeline = RAGPipeline()

        docs = [
            "Drain current equation: Id = μCox(W/L)(Vgs - Vth)^2",
            "Threshold voltage decreases with temperature.",
            "MOSFET operates in saturation when Vds > Vgs - Vth."
        ]

        pipeline.build(docs)

        query = "What is the MOSFET drain current equation?"

        result = pipeline.answer(query)

        print("\nRetrieved Evidence:\n", result["evidence"])
        print("\nResponse:\n", result["response"])
        print("\nSymbolic Validation:\n", result["symbolic_validation"])
        print("\nDimension Validation:\n", result["dimension_validation"])
        print("\nConfidence Score:", result["confidence_score"])
        print("Confidence Level:", result["confidence_label"])
        print("\nNumerical Validation:", result["numerical_validation"])
    except Exception as e:
        print("\nCRITICAL CRASH CAUGHT:")
        traceback.print_exc()