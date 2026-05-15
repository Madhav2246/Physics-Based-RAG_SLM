def build_prompt(query, evidence):

    evidence_block = "\n".join(evidence)

    return f"""
Use ONLY the evidence below to answer.

Evidence:
{evidence_block}

Question:
{query}

Answer in 3-4 sentences maximum.
If equation is required, write clearly.
"""