# Viva Defense Guide: Evaluation Stages & Statistics

This guide is designed to help you and your teammate explain the evaluation pipeline and the statistical math clearly during your final defense. 

---

## Part 1: The Story of the 6 Stages

If a professor asks, *"Why do you have 6 different stages? Isn't that messy?"*

**Your answer:** "It’s not messy; it’s a scientific funnel. We didn’t just want to know *if* the system worked; we needed to prove *why* it worked, step-by-step."

Here is how you explain them:

### Stage 0: Pre-registration
* **What it is:** Writing down what we expected to happen *before* we ran the code.
* **Why it matters:** In machine learning, it's easy to run 100 tests and only report the 1 that worked (p-hacking). Stage 0 proves to the examiners that our evaluation was honest and scientifically rigorous.

### Stage 1: The Main Claim (Physics Correctness)
* **What it is:** We ran our 0.5B system and the 70B system on 100 questions and scored them using the SymPy physics validator (checker v3).
* **Why it matters:** This answers the ultimate question: *Does it work?* It proved that our tiny model beats a massive model on hard questions (+135%, SYS 1.868 vs 70B 0.795, p=0.002 **, large effect). The +60% figure refers to the Medium+Hard combined improvement. Overall, SYS leads 70B 1.305 vs 1.271 (n.s.).

### Stage 2: The Sanity Check (Generation Quality)
* **What it is:** We measured Faithfulness, BERTScore, and ROUGE.
* **Why it matters:** Stage 1 only looked at math. Stage 2 proves the model is actually speaking English properly and not just hallucinating random words next to the math. It proved our system is 5.0× more faithful to the documents than the 70B model.

### Stage 3: The Engine Check (Retrieval Quality)
* **What it is:** We measured if the system was actually finding the right documents (Hit@3, MRR).
* **Why it matters:** If the system failed a question, we needed to know if it was the SLM's fault or the Retriever's fault. Stage 3 proved that our equation extractor compensates for imperfect retrieval.

### Stage 4: Taking it Apart (Ablation)
* **What it is:** We turned off features one by one (turned off RAG, turned off dense retrieval) to see how much the score dropped.
* **Why it matters:** It proves *why* the system works. It proved that the corpus equation prepend was the most important part of the architecture (+0.806 score boost, from 0.555 RAW to 1.361 full system).

### Stage 5: The Filter Check (Validator Power)
* **What it is:** We tested if our deterministic physics validator was actually picking the best answers out of 5 generated options.
* **Why it matters:** It proves that neuro-symbolic filtering is better than just picking an answer randomly.

### Stage 6: The Math Check (Significance Testing)
* **What it is:** We ran statistical math to prove our results weren't just a lucky accident. 

---

## Part 2: Explaining the Statistics (For Human Beings)

Your teammate asked: *"What is Wilcoxon rank and two-tailed?"* 

Here is the exact, simple way to explain these concepts to the committee.

### 1. Why do we need statistics at all?
If your system scores 1.43 and the 70B model scores 1.04, a professor will say: *"Maybe your system just got lucky on these specific 20 questions. If we asked 20 different questions, maybe the 70B would win."*
**Statistics is how we mathematically prove that it wasn't luck.**

### 2. What is a p-value?
* **Simple definition:** The probability that our results were just a fluke.
* **The rule:** If $p < 0.05$ (less than 5%), we declare "Statistical Significance." It means we are 95% confident the result is real. 
* **Our result:** For SYS vs RAW, our $p = 0.00000$. This means there is a 0% chance our architecture's success was an accident.

### 3. What is the "Wilcoxon Signed-Rank Test"?
Usually, people use a "T-test" to compare two scores. But a T-test assumes the data is shaped like a perfect bell curve (Normal Distribution). 
* Our physics scores are bounded between 0 and 4. They are not a bell curve.
* The **Wilcoxon Signed-Rank Test** is a "non-parametric" test. It doesn't care about bell curves. It just ranks the differences between the two systems. 
* **How to say it in the Viva:** *"We used the Wilcoxon test because our 0-4 physics scores are not normally distributed, making standard T-tests mathematically invalid."* (Professors will love this answer).

### 4. What does "Two-Tailed" mean?
* **One-tailed test:** You are only testing if System A is *better* than System B. If System A is secretly worse, the test ignores it. It's considered "cheating" to get a better p-value.
* **Two-tailed test:** You are testing if System A is *different* from System B (it could be better OR worse).
* **How to say it in the Viva:** *"We used a two-tailed test because it is more conservative and mathematically rigorous. We didn't want to bias the statistics in our favor."*

### 5. What is the "Effect Size (r)"?
* The p-value only tells you *if* the difference is real. The **Effect Size (r)** tells you *how big* the difference is.
* Scale: 0.1 is small, 0.3 is medium, 0.5 is large.
* **Our result:** Our effect size for SYS vs RAW was **0.498** (medium), and for the hard-question comparison (T5: SYS vs 70B on hard) it was **r=0.678** (large). Both prove the architecture radically changes the model's behavior — the hard-question effect size is in the large range, meaning the result is not just statistically significant but practically substantial.

---

## Part 3: Quick Q&A for the Defense

**Q: "Why did your system tie the 70B model overall, instead of beating it completely?"**
> **A:** "Our system actually leads 70B overall now (1.305 vs 1.271), but the difference is not statistically significant — it's within noise at n=100. What matters is the difficulty stratification: large models win on easy, rote-memorization questions due to their massive parameter count. But on hard questions, parametric memory fails. Our system explicitly trades memorization for structural reasoning, which is why we achieve a +135% advantage (p=0.002, large effect) on the hardest engineering questions and +60% on medium+hard combined."

**Q: "What exactly does your 'physics validator' do?"**
> **A:** "It treats the LLM's text output as untrusted. It extracts the equation, parses it into an AST using SymPy, checks if the SI units match on both sides, and plugs in actual technology node parameters (like 5nm GAA) to see if the math computes. Only then does it assign a score."

**Q: "Why use 6 stages instead of just showing me a graph?"**
> **A:** "Because an AI system has many moving parts. If we just showed one graph, we wouldn't know *why* it worked. Our stages isolate retrieval (Stage 3), isolate the validator (Stage 5), and isolate the grounding (Stage 4). It is a component-level ablation study."
