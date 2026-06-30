"""
SLMExtractor — natural-language physics parameter extractor.

Uses the already-loaded TinySLM (0.5B) to convert free-form query text into
a structured JSON dict of physics variables, then verifies every extracted
value against tokens actually present in the query (the verification gate).

Architecture boundary (CRITICAL):
  - This component extracts numbers from language. It does NOT compute, derive,
    or substitute any values. All math stays in SymPy (solve_for).
  - If extraction fails for any reason, the caller falls through to the existing
    Regex extractor silently. Never raises, never crashes the pipeline.

Stage 1 (this file):  SLM → JSON → verify against query tokens
Stage 2 (caller):     Regex scan for anything Stage 1 missed

─────────────────────────────────────────────────────────────────────────────
KNOWN LIMITATIONS (as of June 2026)
─────────────────────────────────────────────────────────────────────────────
The TinySLM (0.5B, LoRA fine-tuned on narrative physics Q&A) does not reliably
produce strict JSON when prompted for parameter extraction. At this model size,
instruction-following capability for schema-constrained generation is limited.

Empirical finding (extractor_benchmark.py, June 2026):
  - Regex-only score:     20/22  (after query-5 SLM interference fix)
  - Two-stage score:      20/22  (SLM adds 0 net extractions)
  - SLM Stage 1 returns {} for all 15 benchmark queries in practice.

The score improvement from 6/22 → 19/22 (versus the pre-Feature-3 baseline)
is attributable entirely to the 18 natural-language regex patterns added to
ExplorationEngine.VALUE_PATTERNS in Stage 2, not to the SLM.

The two-stage architecture is correct and future-proof:
  - Swapping the 0.5B model for a dedicated NER model (e.g., BERT-NER fine-tuned
    on physics parameter extraction, ~110M params) would immediately activate
    Stage 1 without any other code changes.
  - The verification gate (_verify_extractable) remains critical to prevent any
    model from hallucinating values not present in the query text.

For the current system, Stage 1 is effectively dormant. Stage 2 (regex) carries
all extractions. This should be stated honestly in any academic report.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import json
import re
import math
import logging

logger = logging.getLogger(__name__)

# Recognised symbol keys the SLM is allowed to output
_VALID_SYMBOLS = {
    "Id", "Vov", "Vgs", "Vth", "Vsb", "gamma",
    "tox", "L", "W", "mu", "Cox", "Phi_f", "Vth0", "T",
}

# Unit-word → SI multiplier (for the verification gate token scan)
_WORD_SCALE = {
    # current
    "picoamp": 1e-12, "picoamps": 1e-12, "pa": 1e-12,
    "nanoamp": 1e-9,  "nanoamps": 1e-9,  "na": 1e-9,
    "microamp": 1e-6, "microamps": 1e-6, "ua": 1e-6, "µa": 1e-6,
    "milliamp": 1e-3, "milliamps": 1e-3, "ma": 1e-3,
    "amp": 1.0, "amps": 1.0, "ampere": 1.0, "amperes": 1.0, "a": 1.0,
    # voltage
    "microvolt": 1e-6, "microvolts": 1e-6, "uv": 1e-6,
    "millivolt": 1e-3, "millivolts": 1e-3, "mv": 1e-3,
    "kilovolt": 1e3,   "kilovolts": 1e3,   "kv": 1e3,
    "volt": 1.0, "volts": 1.0, "v": 1.0,
    # length
    "picometer": 1e-12, "picometers": 1e-12, "pm": 1e-12,
    "nanometer": 1e-9,  "nanometers": 1e-9,  "nm": 1e-9,
    "micrometer": 1e-6, "micrometers": 1e-6, "um": 1e-6, "micron": 1e-6, "microns": 1e-6,
    "millimeter": 1e-3, "millimeters": 1e-3, "mm": 1e-3,
    "meter": 1.0, "meters": 1.0, "m": 1.0,
    # fractions expressed as words
    "half": 0.5, "quarter": 0.25,
    # generic
    "": 1.0,
}

_NUM_PATTERN = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')


def _tokens_from_query(query: str) -> list[float]:
    """
    Return all plausible numeric values that can be inferred from query tokens.
    Handles: digits, decimal, scientific, and word-units like 'milliamp', 'half'.
    """
    values: list[float] = []

    # 1. Explicit numeric tokens (possibly with inline unit)
    # e.g. "1mA", "0.5V", "2e-3A", "100nm"
    for m in re.finditer(
        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*'
        r'(pA|nA|uA|µA|mA|A|uV|mV|kV|V|pm|nm|um|µm|mm|m)?',
        query, re.IGNORECASE
    ):
        num_str, unit = m.group(1), (m.group(2) or "").lower()
        try:
            scale = _WORD_SCALE.get(unit, 1.0)
            values.append(float(num_str) * scale)
        except ValueError:
            pass

    # 2. Word-magnitude combos: "one milliamp", "half a volt", "zero point three"
    # Simple approach: scan for known magnitude words and the number before them
    for word, scale in _WORD_SCALE.items():
        if not word or scale == 1.0:
            continue
        pat = rf'(\d*\.?\d+(?:[eE][-+]?\d+)?)\s+{re.escape(word)}\b'
        for m in re.finditer(pat, query, re.IGNORECASE):
            try:
                values.append(float(m.group(1)) * scale)
            except ValueError:
                pass

    # 3. "half" / "quarter" alone
    if re.search(r'\bhalf\b', query, re.IGNORECASE):
        values.append(0.5)
    if re.search(r'\bquarter\b', query, re.IGNORECASE):
        values.append(0.25)

    return values


def _verify_extractable(value: float, query: str,
                         tol: float = 0.02) -> bool:
    """
    Accept the extracted value only if some numeric token in the query
    is plausibly its source (within tol relative tolerance after unit conversion).

    Returns False if the value cannot be traced to any token — this prevents
    the SLM from fabricating numbers not present in the query text.
    """
    if value == 0.0:
        # Match a standalone zero token ("0", "0V", "0 volts") but NOT a
        # decimal digit sequence like "0.3" where "0" is just the leading digit.
        # Negative lookahead (?!\.\d) ensures we don't match "0.3", "0.5" etc.
        return re.search(r'\b0\b(?!\.\d)', query) is not None

    candidates = _tokens_from_query(query)
    for cand in candidates:
        if cand == 0.0:
            continue
        rel_err = abs(value - cand) / (abs(cand) + 1e-30)
        if rel_err <= tol:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────

class SLMExtractor:
    """
    Stage 1 of the two-stage value extractor.

    Sends a constrained JSON-extraction prompt to the already-loaded SLM,
    parses the output, and verifies each value against the source query.
    Falls back to {} on any failure — never raises.
    """

    EXTRACTION_PROMPT = (
        "You are a physics parameter extractor. "
        "Extract all numeric values from the query below into strict JSON.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON. No preamble, no explanation, no markdown.\n"
        "- Convert all values to SI base units "
        "(mA -> A, nm -> m, uA -> A, milliamp -> 1e-3 A, half -> 0.5).\n"
        "- Only include parameters EXPLICITLY stated in the query.\n"
        "- Do NOT infer, assume, or use defaults for unstated parameters.\n"
        "- Use ONLY these exact symbol keys: "
        "Id, Vov, Vgs, Vth, Vsb, gamma, tox, L, W, mu, Cox, Phi_f\n"
        '- If no values are stated, output: {{}}\n\n'
        'Query: "{query}"\n\n'
        "JSON:"
    )

    def __init__(self, model) -> None:
        """
        model: already-instantiated TinySLM (no new weights loaded).
        Pass None to disable SLM extraction (fallback to regex only).
        """
        self._model = model

    @property
    def available(self) -> bool:
        return self._model is not None

    def extract(self, query: str) -> dict[str, float]:
        """
        Run Stage 1 extraction. Returns {symbol: SI_value} for verified values.
        Returns {} on any failure — caller must then run Stage 2 (regex).
        """
        if not self.available:
            return {}

        prompt = self.EXTRACTION_PROMPT.format(query=query)
        try:
            outputs = self._model.generate_multiple(prompt, n_samples=1, max_tokens=120)
            raw = outputs[0].strip() if outputs else ""
        except Exception as exc:
            logger.warning("[SLMExtractor] generate failed: %s", exc)
            return {}

        # Strip markdown fences the model sometimes adds despite instructions
        raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw).rstrip('`').strip()

        # Find first {...} block
        m = re.search(r'\{[^}]*\}', raw, re.DOTALL)
        if not m:
            logger.debug("[SLMExtractor] no JSON block found in: %r", raw[:200])
            return {}

        try:
            parsed = json.loads(m.group())
        except json.JSONDecodeError as exc:
            logger.debug("[SLMExtractor] JSON parse failed (%s): %r", exc, raw[:200])
            return {}

        # Validate keys and verify against query tokens
        result: dict[str, float] = {}
        for key, val in parsed.items():
            if key not in _VALID_SYMBOLS:
                logger.debug("[SLMExtractor] unknown symbol %r — discarded", key)
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                logger.debug("[SLMExtractor] non-numeric value for %r: %r", key, val)
                continue

            if _verify_extractable(fval, query):
                result[key] = fval
            else:
                logger.debug(
                    "[SLMExtractor] %s=%g not traceable to query — discarded", key, fval
                )

        return result
