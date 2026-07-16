import json
import argparse
from pathlib import Path

def compute_metrics(eval_file: Path, correctness_threshold: float = 0.70):
    if not eval_file.exists():
        print(f"Error: Could not find {eval_file}")
        return

    total = 0
    
    # Categories based on Ground Truth Correctness
    correct_answers = []
    wrong_answers = []

    # Categories based on Model Confidence
    high_conf = []
    mod_conf = []
    low_conf = []

    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            total += 1
            conf_score = entry.get("confidence_score", 0.0)
            semantic_score = entry.get("semantic_similarity", 0.0)
            label = entry.get("confidence_label", "")

            # 1. Define "Correctness" proxy
            # We use semantic similarity > threshold as a proxy for correctness 
            # against the NVIDIA expected answer.
            is_correct = semantic_score >= correctness_threshold
            
            if is_correct:
                correct_answers.append(entry)
            else:
                wrong_answers.append(entry)

            # 2. Bucket by Confidence Label
            if "HIGH" in label:
                high_conf.append(entry)
            elif "MODERATE" in label:
                mod_conf.append(entry)
            else:
                low_conf.append(entry)

    if total == 0:
        print("No evaluation results found.")
        return

    # METRIC 1: Hallucination Catch Rate
    # Of all WRONG answers, how many did the validator catch (LOW or MODERATE)?
    wrong_caught = [ans for ans in wrong_answers if "HIGH" not in ans.get("confidence_label", "")]
    catch_rate = (len(wrong_caught) / len(wrong_answers)) * 100 if wrong_answers else 100.0

    # METRIC 2: False-Confidence Rate
    # Of all HIGH CONFIDENCE answers, how many were actually wrong?
    high_wrong = [ans for ans in high_conf if ans not in correct_answers]
    false_conf_rate = (len(high_wrong) / len(high_conf)) * 100 if high_conf else 0.0

    # METRIC 3: Confidence Separation
    avg_conf_correct = sum(a.get("confidence_score", 0.0) for a in correct_answers) / len(correct_answers) if correct_answers else 0.0
    avg_conf_wrong = sum(a.get("confidence_score", 0.0) for a in wrong_answers) / len(wrong_answers) if wrong_answers else 0.0
    separation = avg_conf_correct - avg_conf_wrong

    print("=" * 60)
    print(f"CALIBRATION METRICS (Threshold: semantic >= {correctness_threshold})")
    print("=" * 60)
    print(f"Total Evaluated:   {total}")
    print(f"Proxy Correct:     {len(correct_answers)}")
    print(f"Proxy Wrong:       {len(wrong_answers)}\n")

    print(f"1. Hallucination Catch Rate: {catch_rate:.1f}%")
    print(f"   ({len(wrong_caught)} / {len(wrong_answers)} wrong answers successfully flagged as LOW/MODERATE)\n")

    print(f"2. False-Confidence Rate:    {false_conf_rate:.1f}%")
    print(f"   ({len(high_wrong)} / {len(high_conf)} HIGH confidence answers were actually wrong)\n")

    print(f"3. Confidence Separation:    +{separation:.3f}")
    print(f"   (Correct Avg: {avg_conf_correct:.3f} | Wrong Avg: {avg_conf_wrong:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute calibration metrics for the SLM Paper")
    parser.add_argument("--eval_file", type=str, default="data/evaluation/eval_results.jsonl")
    parser.add_argument("--threshold", type=float, default=0.70, help="Semantic similarity threshold to count as 'correct'")
    args = parser.parse_args()
    
    compute_metrics(Path(args.eval_file), args.threshold)
