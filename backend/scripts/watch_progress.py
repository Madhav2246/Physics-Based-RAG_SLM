import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
LIVE_FILE     = PROJECT_ROOT / "data" / "evaluation" / "live.json"

R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
G   = "\033[92m"
Y   = "\033[93m"
M   = "\033[95m"
C   = "\033[96m"
RE  = "\033[91m"

DIFF_COLOR = {"easy": G, "medium": Y, "hard": M}
SPINNER    = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]


def bar(done, total, width=28, col=C):
    filled = int(width * done / total) if total else 0
    return f"{col}{'#' * filled}{DIM}{'-' * (width - filled)}{R}"


def render(data, tick):
    os.system("cls" if os.name == "nt" else "clear")

    status   = data.get("status", "waiting")
    updated  = data.get("updated_at", "")
    overall  = data.get("overall",       {"done": 0, "target": 100, "percent": 0.0})
    by_diff  = data.get("by_difficulty", {})
    last     = data.get("last",          {})

    done    = overall["done"]
    target  = overall["target"]
    percent = overall["percent"]

    spin = "✅" if status == "complete" else SPINNER[tick % 10]

    print(f"\n  {B}{C}Physics-RAG  ·  Dataset Synthesis{R}  {DIM}(live.json){R}\n")

    # overall bar
    print(f"  {spin}  {bar(done, target, width=38, col=C)}  "
          f"{C}{B}{done}/{target}{R}  {percent}%")

    # per-difficulty rows
    print()
    for diff in ["easy", "medium", "hard"]:
        d   = by_diff.get(diff, {})
        col = DIFF_COLOR.get(diff, C)
        chk = f"{G}[OK]{R}" if d.get("complete") else " "
        print(f"  {chk} {col}{B}{diff:<6}{R}  "
              f"{bar(d.get('done',0), d.get('target',1), width=24, col=col)}  "
              f"{col}{d.get('done',0)}/{d.get('target',0)}{R}  "
              f"{DIM}{d.get('remaining',0)} left{R}")

    # last generated question
    if last.get("question"):
        q   = last["question"]
        qid = last.get("id", "")
        col = DIFF_COLOR.get(last.get("difficulty", ""), C)
        print(f"\n  {DIM}last ·{R} {col}{B}[{qid}]{R}  "
              f"{q[:70]}{'…' if len(q) > 70 else ''}")

    print(f"\n  {DIM}{updated}   ctrl+c to exit{R}\n")


def main():
    tick = 0
    print(f"  watching {LIVE_FILE} …")
    while True:
        try:
            if LIVE_FILE.exists():
                data = json.loads(LIVE_FILE.read_text(encoding="utf-8"))
                render(data, tick)
                if data.get("status") == "complete":
                    print("  done.\n")
                    break
            else:
                os.system("cls" if os.name == "nt" else "clear")
                print(f"\n  {Y}waiting for synthesize_data.py to start…{R}\n"
                      f"  {DIM}(will appear at {LIVE_FILE}){R}\n")
            tick += 1
            time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n  {DIM}watcher stopped.{R}\n")
            break
        except json.JSONDecodeError:
            tick += 1
            time.sleep(2)

if __name__ == "__main__":
    main()
