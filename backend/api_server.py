"""
Physics RAG SLM — FastAPI server

Run from the backend/ directory:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Endpoints consumed by the Lovable frontend (frontend/src/lib/api.ts → API_BASE = http://localhost:8000):

    GET  /                            health + pipeline_ready flag
    POST /api/query                   RAG inference (returns full inspector data)
    POST /api/query/baseline          bare Qwen inference (no retrieval, no validation)
    GET  /api/registry                knowledge base document list + stats
    POST /api/ingest                  upload PDFs and run ingestion pipeline
    POST /api/ingest/reset            wipe FAISS + BM25 indexes
    GET  /api/synthesis/live          live progress of synthesize_data.py
    POST /api/synthesis/start         start synthesize_data.py as a subprocess
    GET  /api/evaluation/live         live progress of evaluate_pipeline.py
    POST /api/evaluation/start        start evaluate_pipeline.py as a subprocess
    POST /api/feedback                write a HITL correction/gap/thumbs-up to JSONL
    GET  /api/feedback/count          count of pending correction pairs
    GET  /api/train/live              live training log from train_from_feedback.py
    POST /api/train/from-feedback     start train_from_feedback.py as a subprocess
"""

import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Root = backend/ ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)          # all relative paths in utils/config.py resolve here
sys.path.insert(0, str(PROJECT_ROOT))

# Remove broken corporate SSL cert if the file no longer exists
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]
os.environ["HF_HOME"] = "d:/S6/NLP/Physics_Based_RAG_SLM/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# --- Paths (mirrors utils/config.py relative paths) ---------------------------
DATA_DIR      = PROJECT_ROOT / "data"
EMB_DIR       = DATA_DIR / "embeddings"
EVAL_DIR      = DATA_DIR / "evaluation"
FEEDBACK_DIR  = DATA_DIR / "feedback"
LOG_FILE      = PROJECT_ROOT / "logs" / "rag_logs.jsonl"
REGISTRY_FILE = EMB_DIR / "registry.json"
BM25_FILE     = EMB_DIR / "bm25_docs.json"
SYNTH_LIVE    = EVAL_DIR / "live.json"
EVAL_LIVE     = EVAL_DIR / "live_evaluation_nvidia_golden_qa.json"
TRAIN_LIVE    = FEEDBACK_DIR / "live_training.json"

# --- NVIDIA NIM config --------------------------------------------------------
# Key is read from the environment; the literal below is only a last-resort
# fallback so the server starts in dev without extra setup.
NVIDIA_API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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",
)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# llama-3.3-70b is the maintained successor of 3.1-70b on NIM (same interface,
# 3.1 is being sunset). It is the strongest defensible "big model" baseline:
# strong enough to stress-test our validator, still fast and cheap on NIM.
BASELINE_MODEL = "meta/llama-3.3-70b-instruct"
CORRECTIONS   = FEEDBACK_DIR / "hitl_corrections.jsonl"

# --- App + CORS ---------------------------------------------------------------
app = FastAPI(title="Physics RAG SLM API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated artifacts (sweep PNGs) so the frontend can <img src=...> them.
from fastapi.staticfiles import StaticFiles
EVAL_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static/evaluation", StaticFiles(directory=str(EVAL_DIR)), name="evaluation")

# --- Pipeline singleton -------------------------------------------------------
_pipeline = None
_pipeline_ready = False
_executor = ThreadPoolExecutor(max_workers=2)


def _load_pipeline():
    global _pipeline, _pipeline_ready
    print("[API] Loading RAG pipeline (this takes ~30s on first run)…")
    try:
        from pipeline.rag_pipeline import RAGPipeline
        p = RAGPipeline()
        p.retriever.dense.load_index()
        p.retriever.sparse.build_index_from_docs(p.retriever.dense.documents)
        _pipeline = p
        _pipeline_ready = True
        print("[API] Pipeline ready.")
    except Exception as e:
        print("[API] ERROR LOADING PIPELINE:")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _load_pipeline)


# --- Pydantic models ----------------------------------------------------------
class QueryBody(BaseModel):
    question: str


class FeedbackBody(BaseModel):
    question: str
    bad_response: str
    feedback_type: str
    correct_response: Optional[str] = None
    missing_topic: Optional[str] = None


class SynthStartBody(BaseModel):
    target_count: int = 100


class EvalStartBody(BaseModel):
    dataset: str = "data/evaluation/nvidia_golden_qa.jsonl"


class TrainBody(BaseModel):
    learning_rate: float = 2e-4
    lora_rank: int = 16
    batch_size: int = 1
    max_steps: int = 200
    warmup_steps: int = 10
    lr_scheduler: str = "cosine"


# --- Helpers ------------------------------------------------------------------
def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _format_breakdown(pipeline_breakdown: list) -> list:
    """
    Converts the pipeline's confidence_breakdown into the shape the frontend expects.
    Uses real values from ConfidenceEngine.compute() — nothing is hardcoded here.
    """
    return [
        {
            "label": item["label"],
            "value": item["score"],   # actual points earned, not max
            "max":   item["max"],
            "ok":    item["ok"],
        }
        for item in pipeline_breakdown
    ]


def _make_explanation(symbolic: str, dimensional: str, numerical: str,
                      equation: str = "", model_label: str = "RAG 0.5B") -> dict:
    """Generate human-readable physics checker explanations for the Inspector."""
    from physics.physics_explainer import explain_compact
    return explain_compact(symbolic, dimensional, numerical, equation, model_label)


def _format_chunks(evidence: list) -> list:
    return [{"source": f"Chunk {i+1} · FAISS+BM25→CrossEncoder", "text": chunk}
            for i, chunk in enumerate(evidence)]


def _sweep_plot_url(plot_path: str | None) -> str | None:
    """
    Map an absolute sweep PNG path under data/evaluation/ to a static URL the
    frontend can load. Returns None if no plot or the path is outside EVAL_DIR.
    """
    if not plot_path:
        return None
    try:
        rel = Path(plot_path).resolve().relative_to(EVAL_DIR.resolve())
        return f"/static/evaluation/{rel.as_posix()}"
    except Exception:
        return None


def _serialize_explore_result(er: dict | None) -> dict | None:
    """
    Convert the raw explore_result into a JSON-safe dict for the frontend.
    The raw dict holds a ValueTracker and SymPy objects that FastAPI cannot
    encode — strip those, keep only display-ready primitives.
    """
    if not er:
        return None

    tracker = er.get("tracker")
    provenance = []
    provenance_fraction = None
    if tracker is not None:
        try:
            provenance_fraction = round(tracker.provenance_fraction, 3)
            for tv in tracker._values.values():
                provenance.append({
                    "symbol":      tv.symbol,
                    "value":       tv.value,
                    "unit":        tv.unit,
                    "provenance":  tv.provenance,   # user | corpus | default
                    "description": tv.description,
                })
        except Exception:
            pass

    # Sweep curve (only present in SWEEP mode)
    sweep = None
    sr = er.get("sweep_result")
    if sr is not None and not getattr(sr, "error", ""):
        sweep = {
            "sweep_var":  sr.sweep_var,
            "target_var": sr.target_var,
            "x":          sr.x,
            "y":          sr.y,
            "node_name":  sr.node_name,
        }

    return {
        "success":         er.get("success", False),
        "target":          er.get("target"),
        "symbolic":        er.get("symbolic"),          # str form, e.g. "WL = 2*Id/..."
        "numeric":         er.get("numeric"),
        "sanity_ok":       er.get("sanity_ok"),
        "error":           er.get("error", ""),
        "corpus_equation": er.get("corpus_equation"),
        "provenance":      provenance,
        "provenance_fraction": provenance_fraction,
        "sweep":           sweep,
    }


# --- Health -------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "pipeline_ready": _pipeline_ready,
        "model": "Qwen/Qwen2.5-0.5B-Instruct + LoRA",
    }


# --- Query: RAG ---------------------------------------------------------------
@app.post("/api/query")
async def rag_query(body: QueryBody):
    if not _pipeline_ready:
        return {"error": "Pipeline loading — please wait ~30s.", "pipeline_ready": False}

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _pipeline.answer, body.question)

    sym_msg = result["symbolic_validation"]
    dim_msg = result["dimension_validation"]
    num_msg = result["numerical_validation"]

    # Use the breakdown the pipeline already computed — no re-parsing strings here
    breakdown = _format_breakdown(result["confidence_breakdown"])

    # Extract the equation string for the explainer
    from physics.equation_validator import EquationValidator
    _eq_line = EquationValidator().extract_equation(result["response"]) or ""

    explanation = _make_explanation(sym_msg, dim_msg, num_msg, _eq_line, "RAG 0.5B")

    return {
        "response":    result["response"],
        "confidence":  result["confidence_score"],
        "breakdown":   breakdown,
        "symbolic":    sym_msg,
        "dimensional": dim_msg,
        "numerical":   num_msg,
        "stability": {
            "score": result["uncertainty_score"],
            "label": result["stability_label"],
        },
        "similarity":     result["semantic_similarity"] or 0.0,
        "chunks":         _format_chunks(result["evidence"]),
        "explanation":    explanation,
        "mode":           result.get("mode", "LOOKUP"),
        "explore_result": _serialize_explore_result(result.get("explore_result")),
        "node_profile":   result.get("node_profile", "100nm_CMOS (default)"),
        # Relative URL the frontend can load as an <img>; None unless SWEEP succeeded
        "sweep_plot_url": _sweep_plot_url(result.get("sweep_plot_path")),
    }


# --- Query: Baseline (strong NVIDIA NIM model — no retrieval, no validation) --
@app.post("/api/query/baseline")
def baseline_query(body: QueryBody):
    """
    Runs the question against a large NVIDIA NIM model with NO physics
    retrieval and NO deterministic validation.  The purpose is to showcase
    that even a 70B SOTA model produces physics-invalid equations that our
    validator would catch — not to claim our 0.5B beats it on answer quality.
    """
    from openai import OpenAI
    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        completion = client.chat.completions.create(
            model=BASELINE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a semiconductor device physics expert. "
                        "Answer the question concisely and include the key equation. "
                        "Do NOT use retrieval — rely only on your parametric knowledge."
                    ),
                },
                {"role": "user", "content": body.question},
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=512,
        )
        response_text = completion.choices[0].message.content
        
        # Run physics validation on baseline to prove it fails
        from physics.equation_validator import EquationValidator
        from physics.dimension_checker import DimensionChecker
        from physics.numerical_validator import NumericalValidator
        
        validator = EquationValidator()
        lhs, rhs, sym_msg = validator.validate(response_text)
        dim_msg = DimensionChecker().check_equation(lhs, rhs) if lhs else "[WARN] Dimension check skipped."
        num_msg = NumericalValidator().evaluate(lhs, rhs) if rhs else "[WARN] Numerical check skipped."
        eq_line = validator.extract_equation(response_text) or ""
        explanation = _make_explanation(sym_msg, dim_msg, num_msg, eq_line, BASELINE_MODEL)

        # Compute confidence score using the shared engine
        from utils.confidence_engine import ConfidenceEngine
        confidence_engine = ConfidenceEngine()
        confidence_score, _ = confidence_engine.compute(
            [],
            sym_msg,
            dim_msg,
            response_text,
            num_msg,
            semantic_similarity=None,
        )

        return {
            "response": response_text,
            "model": BASELINE_MODEL,
            "explanation": explanation,
            "symbolic": sym_msg,
            "dimensional": dim_msg,
            "numerical": num_msg,
            "confidence": confidence_score
        }
    except Exception as e:
        return {"response": f"NVIDIA Baseline Error: {str(e)}", "model": BASELINE_MODEL}


# --- Query: RAG model size comparison (0.5B live + 1.5B/3B via NIM) ----------
_NIM_RAG_SYSTEM = (
    "You are a semiconductor device physics assistant. "
    "The following corpus evidence was retrieved from a physics textbook. "
    "Use it to answer the question. Include the key equation if present, "
    "then explain each symbol in a plain bulleted list. "
    "Use plain text only — NO LaTeX, NO \\[, NO \\(, NO dollar signs."
)

def _call_nim_rag(client, nim_model: str, question: str, evidence: list[str]) -> dict:
    """
    Call a NIM-hosted model with the retrieved evidence injected as context.
    Returns the same physics-validated shape as the main /api/query endpoint.
    """
    from physics.equation_validator import EquationValidator
    from physics.dimension_checker import DimensionChecker
    from physics.numerical_validator import NumericalValidator

    evidence_block = "\n".join(f"- {e[:300]}" for e in evidence[:3])  # keep prompt short
    user_msg = f"Evidence:\n{evidence_block}\n\nQuestion: {question}"

    try:
        completion = client.chat.completions.create(
            model=nim_model,
            messages=[
                {"role": "system", "content": _NIM_RAG_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=512,
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        response_text = f"NIM call failed: {str(e)}"

    validator = EquationValidator()
    lhs, rhs, sym_msg = validator.validate(response_text)
    dim_msg = DimensionChecker().check_equation(lhs, rhs) if lhs else "[WARN] Dimension check skipped."
    num_msg = NumericalValidator().evaluate(lhs, rhs) if rhs else "[WARN] Numerical check skipped."
    eq_line = validator.extract_equation(response_text) or ""
    explanation = _make_explanation(sym_msg, dim_msg, num_msg, eq_line, nim_model)

    return {
        "response":    response_text,
        "symbolic":    sym_msg,
        "dimensional": dim_msg,
        "numerical":   num_msg,
        "explanation": explanation,
    }


@app.post("/api/query/rag-compare")
async def rag_compare(body: QueryBody):
    """
    Fires the same question through all three RAG model sizes simultaneously:
      - 0.5B  : local Qwen LoRA (already computed by /api/query — caller passes it)
      - 1.5B  : Qwen2.5-1.5B-Instruct on NVIDIA NIM + retrieved evidence
      - 3B    : Qwen2.5-3B-Instruct on NVIDIA NIM + retrieved evidence

    The 0.5B result is NOT re-computed here to avoid duplicating model load;
    the frontend should already have it from /api/query.
    """
    if not _pipeline_ready:
        return {"error": "Pipeline loading — please wait ~30s."}

    from openai import OpenAI
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)

    # Retrieve evidence once, share it across all model calls
    loop = asyncio.get_event_loop()
    evidence = await loop.run_in_executor(
        _executor,
        lambda: _pipeline.retriever.retrieve(body.question, top_k=3)
    )

    # Call 1.5B and 3B concurrently via NIM
    nim_1_5b = "qwen/qwen2.5-1.5b-instruct"
    nim_3b   = "qwen/qwen2.5-3b-instruct"

    r15, r3 = await asyncio.gather(
        loop.run_in_executor(_executor, _call_nim_rag, client, nim_1_5b, body.question, evidence),
        loop.run_in_executor(_executor, _call_nim_rag, client, nim_3b,   body.question, evidence),
    )

    return {
        "evidence_chunks": _format_chunks(evidence),
        "model_1_5b": {**r15, "model": nim_1_5b},
        "model_3b":   {**r3,  "model": nim_3b},
    }


# --- Registry -----------------------------------------------------------------
@app.get("/api/registry")
def registry():
    reg = _read_json(REGISTRY_FILE, {})

    documents = [
        {
            "file_name":   v["original_name"],
            "chunks":      v["chunk_count"],
            "faiss_start": v["chunk_start"],
            "ingested_at": v["ingested_at"],
        }
        for v in reg.values()
    ]

    total_pdfs    = len(reg)
    faiss_vectors = sum(v["chunk_count"] for v in reg.values())

    bm25_vocab = 0
    if BM25_FILE.exists():
        try:
            token_lists = json.loads(BM25_FILE.read_text(encoding="utf-8"))
            bm25_vocab  = len({tok for lst in token_lists for tok in lst})
        except Exception:
            pass

    return {
        "documents": documents,
        "stats": {
            "total_pdfs":    total_pdfs,
            "faiss_vectors": faiss_vectors,
            "bm25_vocab":    bm25_vocab,
        },
    }


# --- Ingest -------------------------------------------------------------------
@app.post("/api/ingest")
async def ingest_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    tmp = Path(tempfile.mkdtemp())
    pdf_paths = []
    for f in files:
        dest = tmp / f.filename
        dest.write_bytes(await f.read())
        pdf_paths.append(dest)

    filenames = [f.filename for f in files]

    def _run():
        from scripts._ingestion_engine import IngestionEngine
        IngestionEngine().ingest_pdfs(pdf_paths)
        shutil.rmtree(tmp, ignore_errors=True)

    background_tasks.add_task(lambda: _executor.submit(_run))
    return {"status": "ingestion_started", "files": filenames}


@app.post("/api/ingest/reset")
def ingest_reset():
    from scripts._ingestion_engine import IngestionEngine
    IngestionEngine().reset()
    return {"status": "reset"}


# --- Synthesis ----------------------------------------------------------------
_synth_proc: Optional[subprocess.Popen] = None


@app.get("/api/synthesis/live")
def synthesis_live():
    raw = _read_json(SYNTH_LIVE, {})
    if not raw:
        return {
            "status": "idle", "overall": 0, "target": 100,
            "easy": {"done": 0, "total": 40},
            "medium": {"done": 0, "total": 40},
            "hard": {"done": 0, "total": 20},
            "last_question": "",
        }
    bd = raw.get("by_difficulty", {})
    return {
        "status":        raw.get("status", "idle"),
        "overall":       raw.get("overall", {}).get("done", 0),
        "target":        raw.get("overall", {}).get("target", 100),
        "easy":          {"done": bd.get("easy",   {}).get("done", 0), "total": 40},
        "medium":        {"done": bd.get("medium", {}).get("done", 0), "total": 40},
        "hard":          {"done": bd.get("hard",   {}).get("done", 0), "total": 20},
        "last_question": raw.get("last", {}).get("question", ""),
    }


@app.post("/api/synthesis/start")
def synthesis_start(_body: SynthStartBody, background_tasks: BackgroundTasks):
    global _synth_proc

    def _run():
        global _synth_proc
        _synth_proc = subprocess.Popen(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "synthesize_data.py")],
            cwd=str(PROJECT_ROOT),
        )
        _synth_proc.wait()

    background_tasks.add_task(lambda: _executor.submit(_run))
    return {"status": "started"}


# --- Evaluation ---------------------------------------------------------------
_eval_proc: Optional[subprocess.Popen] = None


@app.get("/api/evaluation/live")
def evaluation_live():
    raw = _read_json(EVAL_LIVE, {})
    if not raw:
        return {"status": "idle", "done": 0, "total": 0,
                "high_conf": 0, "low_conf": 0, "last_question": ""}
    metrics = raw.get("metrics", {})
    overall = raw.get("overall", {})
    return {
        "status":        raw.get("status", "idle"),
        "done":          overall.get("done", 0),
        "total":         overall.get("target", 0),
        "high_conf":     metrics.get("high_conf_correct", 0),
        "low_conf":      metrics.get("low_conf_caught", 0),
        "last_question": raw.get("last", {}).get("question", ""),
    }


@app.post("/api/evaluation/start")
def evaluation_start(body: EvalStartBody, background_tasks: BackgroundTasks):
    global _eval_proc

    def _run():
        global _eval_proc
        _eval_proc = subprocess.Popen(
            [sys.executable,
             str(PROJECT_ROOT / "scripts" / "evaluate_pipeline.py"),
             "--dataset", body.dataset],
            cwd=str(PROJECT_ROOT),
        )
        _eval_proc.wait()

    background_tasks.add_task(lambda: _executor.submit(_run))
    return {"status": "started"}


# --- Feedback -----------------------------------------------------------------
@app.post("/api/feedback")
def feedback(body: FeedbackBody):
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "question":         body.question,
        "bad_response":     body.bad_response,
        "feedback_type":    body.feedback_type,
        "correct_response": body.correct_response,
        "missing_topic":    body.missing_topic,
        "submitted_at":     datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(CORRECTIONS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "recorded"}


@app.get("/api/feedback/count")
def feedback_count():
    if not CORRECTIONS.exists():
        return {"count": 0}
    lines = [l for l in CORRECTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Only count 'correction' type — those are the ones train_from_feedback.py uses
    count = sum(
        1 for l in lines
        if json.loads(l).get("feedback_type") == "correction"
    )
    return {"count": count}


# --- Training -----------------------------------------------------------------
_train_proc: Optional[subprocess.Popen] = None


@app.get("/api/train/live")
def train_live():
    raw = _read_json(TRAIN_LIVE, {})
    if not raw:
        return {"status": "idle", "step": 0, "max_steps": 0, "loss": None, "logs": []}
    return {
        "status":    raw.get("status", "idle"),
        "step":      raw.get("step", 0),
        "max_steps": raw.get("max_steps", 0),
        "loss":      raw.get("loss"),
        "logs":      raw.get("log", []),
    }


@app.post("/api/train/from-feedback")
def train_from_feedback(body: TrainBody, background_tasks: BackgroundTasks):
    global _train_proc

    def _run():
        global _train_proc
        _train_proc = subprocess.Popen(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_from_feedback.py"),
                "--learning_rate", str(body.learning_rate),
                "--lora_rank",     str(body.lora_rank),
                "--batch_size",    str(body.batch_size),
                "--max_steps",     str(body.max_steps),
                "--warmup_steps",  str(body.warmup_steps),
                "--lr_scheduler",  body.lr_scheduler,
            ],
            cwd=str(PROJECT_ROOT),
        )
        _train_proc.wait()

    background_tasks.add_task(lambda: _executor.submit(_run))
    return {"status": "started"}


# --- Evaluation results (real data for chart) ---------------------------------
@app.get("/api/evaluation/results")
def evaluation_results():
    """
    Reads the latest eval_results_*.jsonl in data/evaluation/ and returns
    a binned confidence-score histogram (20 bins) plus summary stats.
    The frontend replaces its mock bell curve with this real distribution.
    """
    # Pick the most recently modified results file
    candidates = sorted(
        EVAL_DIR.glob("eval_results_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"bins": [], "summary": {"total": 0, "high_conf": 0, "low_conf": 0, "mean": None}}

    scores: list[float] = []
    high_conf = 0
    low_conf  = 0

    try:
        for line in candidates[0].read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                s = rec.get("confidence_score")
                if s is not None:
                    scores.append(float(s))
                    if float(s) >= 0.8:
                        high_conf += 1
                    elif float(s) < 0.55:
                        low_conf += 1
            except Exception:
                pass
    except Exception as e:
        return {"bins": [], "summary": {"error": str(e)}}

    # Build 20 equal-width bins from 0.0 to 1.0
    import math
    n_bins = 20
    bins = [
        {"score": round(i / n_bins, 2), "count": 0}
        for i in range(n_bins)
    ]
    for s in scores:
        idx = min(int(math.floor(s * n_bins)), n_bins - 1)
        bins[idx]["count"] += 1

    mean_score = round(sum(scores) / len(scores), 3) if scores else None

    return {
        "bins": bins,
        "source_file": candidates[0].name,
        "summary": {
            "total":     len(scores),
            "high_conf": high_conf,
            "low_conf":  low_conf,
            "mean":      mean_score,
        },
    }


# --- Feedback corrections preview (before/after for Tuning page) -------------
@app.get("/api/feedback/corrections")
def feedback_corrections(limit: int = 10):
    """
    Returns the most recent 'correction' type feedback entries stored in
    hitl_corrections.jsonl.  Each entry contains:
      - question
      - bad_response  (what the model said before)
      - correct_response (what the user supplied)
      - submitted_at
    This is used by the Tuning page to show the before/after loop is working
    WITHOUT needing to run live inference or reload the LoRA adapter.
    """
    if not CORRECTIONS.exists():
        return {"corrections": [], "total": 0}

    entries = []
    try:
        for line in CORRECTIONS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("feedback_type") == "correction":
                    entries.append({
                        "question":         rec.get("question", ""),
                        "bad_response":     rec.get("bad_response", ""),
                        "correct_response": rec.get("correct_response", ""),
                        "submitted_at":     rec.get("submitted_at", ""),
                    })
            except Exception:
                pass
    except Exception:
        pass

    # newest first
    entries = list(reversed(entries))[:limit]
    return {"corrections": entries, "total": len(entries)}


# --- Entry point --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
