def build_prompt(query, evidence, corpus_equation=None):
    evidence_block = "\n".join(f"- {e}" for e in evidence)

    if corpus_equation:
        # Equation was extracted directly from corpus — model only explains symbols.
        # This is a much simpler task for a 0.5B model.
        return f"""You are a semiconductor device physics assistant.

Evidence:
{evidence_block}

Question: {query}

Equation: {corpus_equation}

Explain what each symbol in the equation above means, using the Evidence.
Use plain text only — NO LaTeX, NO \\[, NO \\(, NO dollar signs.
Format: one bullet point per symbol, e.g.:
- Id = drain current in Amperes
- Cox = oxide capacitance per unit area in F/m2

Explanation:
"""
    else:
        # No clean equation found in corpus — ask model to do its best.
        return f"""You are a semiconductor device physics assistant.

Evidence:
{evidence_block}

Question: {query}

Instructions:
1. Write the key equation using = notation if you can find it in the Evidence.
2. Explain each symbol in one sentence.
3. If no equation is available, explain the concept from the Evidence.

Answer:
"""
