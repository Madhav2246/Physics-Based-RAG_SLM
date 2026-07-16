import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState, useCallback } from "react";
import { Play, Sparkles, FlaskConical, CheckCircle2, Loader2, Database, TrendingUp, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/evaluation")({
  head: () => ({ meta: [{ title: "Evaluation · Physics RAG SLM" }] }),
  component: EvaluationPage,
});

type SynthState = {
  status: "idle" | "starting" | "generating" | "complete" | "error";
  overall: number;
  target: number;
  easy: { done: number; total: number };
  medium: { done: number; total: number };
  hard: { done: number; total: number };
  last_question?: string;
};
type EvalState = {
  status: "idle" | "starting" | "evaluating" | "complete" | "error";
  done: number;
  total: number;
  high_conf: number;
  low_conf: number;
  last_question?: string;
};

const MOCK_SYNTH: SynthState = {
  status: "complete",
  overall: 100,
  target: 100,
  easy: { done: 40, total: 40 },
  medium: { done: 40, total: 40 },
  hard: { done: 20, total: 20 },
  last_question:
    "Derive the threshold voltage shift due to body bias in a long-channel NMOS transistor.",
};
const MOCK_EVAL: EvalState = {
  status: "idle",
  done: 0,
  total: 0,
  high_conf: 0,
  low_conf: 0,
  last_question: "",
};

type EvalResults = {
  bins: { score: number; count: number }[];
  source_file?: string;
  summary: {
    total: number;
    high_conf: number;
    low_conf: number;
    mean: number | null;
    error?: string;
  };
};

function EvaluationPage() {
  const [synth, setSynth] = useState<SynthState>(MOCK_SYNTH);
  const [evalState, setEvalState] = useState<EvalState>(MOCK_EVAL);
  const [target, setTarget] = useState(100);
  const [evalResults, setEvalResults] = useState<EvalResults | null>(null);
  const [loadingResults, setLoadingResults] = useState(false);

  const fetchResults = useCallback(async () => {
    setLoadingResults(true);
    try {
      const r = await apiFetch<EvalResults>("/api/evaluation/results", { timeoutMs: 8000 });
      if (r.bins && r.bins.length > 0) setEvalResults(r);
    } catch {}
    finally { setLoadingResults(false); }
  }, []);

  // Load real results on mount
  useEffect(() => { fetchResults(); }, [fetchResults]);

  // Poll live status + re-fetch results 3 s after eval completes
  useEffect(() => {
    const id = setInterval(() => {
      if (["generating", "starting"].includes(synth.status)) {
        apiFetch<SynthState>("/api/synthesis/live", { timeoutMs: 3000 })
          .then(setSynth)
          .catch(() => {});
      }
      if (["evaluating", "starting"].includes(evalState.status)) {
        apiFetch<EvalState>("/api/evaluation/live", { timeoutMs: 3000 })
          .then((s) => {
            setEvalState(s);
            // When evaluation just completed, refresh the histogram
            if (s.status === "complete") setTimeout(fetchResults, 3000);
          })
          .catch(() => {});
      }
    }, 3000);
    return () => clearInterval(id);
  }, [synth.status, evalState.status, fetchResults]);

  async function startSynth() {
    if (synth.status !== "idle" && synth.status !== "complete") return;
    setSynth({ ...synth, status: "starting" });
    toast.success("Dataset generation started", { description: "NVIDIA API · nvidia_golden_qa.jsonl" });
    try {
      await apiFetch("/api/synthesis/start", {
        method: "POST",
        body: JSON.stringify({ target_count: target }),
      });
    } catch {}
  }

  async function startEval() {
    if (evalState.status !== "idle" && evalState.status !== "complete") return;
    setEvalState({ ...evalState, status: "starting" });
    toast.success("Evaluation started", { description: "Running RAG over nvidia_golden_qa.jsonl" });
    try {
      await apiFetch("/api/evaluation/start", {
        method: "POST",
        body: JSON.stringify({ dataset: "data/evaluation/nvidia_golden_qa.jsonl" }),
      });
    } catch {}
  }

  const synthBusy = synth.status === "starting" || synth.status === "generating";
  const evalBusy = evalState.status === "starting" || evalState.status === "evaluating";

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Evaluation</h1>
        <p className="text-sm text-muted-foreground">
          Generate a golden QA dataset, then benchmark the RAG pipeline against it.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Synthesis */}
        <section className="glass rounded-2xl">
          <div className="flex items-center gap-3 border-b border-border/60 p-4">
            <Sparkles className="h-4 w-4 text-neon" />
            <div>
              <div className="text-sm font-semibold">Golden QA Dataset</div>
              <div className="text-[11px] text-muted-foreground">NVIDIA API · difficulty-balanced</div>
            </div>
          </div>
          <div className="space-y-5 p-5">
            <div className="grid grid-cols-3 gap-2 text-center text-[11px]">
              {[
                { label: "Easy", pct: "40%", color: "bg-success/20 text-success" },
                { label: "Medium", pct: "40%", color: "bg-warning/20 text-warning" },
                { label: "Hard", pct: "20%", color: "bg-destructive/20 text-destructive" },
              ].map((d) => (
                <div key={d.label} className={cn("rounded-lg px-3 py-2 font-medium", d.color)}>
                  {d.label} · {d.pct}
                </div>
              ))}
            </div>

            <div className="flex items-end gap-3">
              <div className="flex-1">
                <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                  Target Count
                </label>
                <Input
                  type="number"
                  value={target}
                  onChange={(e) => setTarget(Number(e.target.value) || 100)}
                  min={10}
                  className="mt-1"
                />
              </div>
              <Button
                onClick={startSynth}
                disabled={synthBusy}
                className="bg-neon text-primary-foreground hover:bg-neon/90 neon-glow"
              >
                {synthBusy ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Running…
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" /> Generate Dataset
                  </>
                )}
              </Button>
            </div>

            <div className="space-y-3">
              <ProgRow
                label="Overall"
                done={synth.overall}
                total={synth.target}
                accent
                done_icon={synth.overall >= synth.target ? "✅" : ""}
              />
              <ProgRow label="Easy" done={synth.easy.done} total={synth.easy.total} />
              <ProgRow label="Medium" done={synth.medium.done} total={synth.medium.total} />
              <ProgRow label="Hard" done={synth.hard.done} total={synth.hard.total} />
            </div>

            {synth.last_question && (
              <div className="rounded-lg border border-border/60 bg-background/30 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Last generated
                </div>
                <div className="mt-1 line-clamp-2 text-xs italic text-foreground/80">
                  “{synth.last_question}”
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Eval */}
        <section className="glass rounded-2xl">
          <div className="flex items-center gap-3 border-b border-border/60 p-4">
            <FlaskConical className="h-4 w-4 text-neon" />
            <div>
              <div className="text-sm font-semibold">RAG Evaluation</div>
              <div className="text-[11px] text-muted-foreground">golden_qa.jsonl</div>
            </div>
          </div>
          <div className="space-y-5 p-5">
            <div className="flex justify-end">
              <Button
                onClick={startEval}
                disabled={evalBusy}
                className="bg-neon text-primary-foreground hover:bg-neon/90 neon-glow"
              >
                {evalBusy ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Evaluating…
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" /> Run Evaluation
                  </>
                )}
              </Button>
            </div>

            <ProgRow label="Progress" done={evalState.done} total={evalState.total} accent />

            <div className="grid grid-cols-2 gap-3">
              <MiniStat
                label="High Confidence Answers"
                value={evalState.high_conf}
                color="text-success"
              />
              <MiniStat
                label="Low Confidence Caught"
                value={evalState.low_conf}
                color="text-warning"
              />
            </div>

            {evalState.last_question && (
              <div className="rounded-lg border border-border/60 bg-background/30 p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
                  Last question
                </div>
                <div className="mt-1 line-clamp-2 text-xs italic text-foreground/80">
                  “{evalState.last_question}”
                </div>
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Results Chart — real data from eval_results_*.jsonl */}
      <section className="glass rounded-2xl p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold">Confidence Score Distribution</h2>
            <p className="text-[11px] text-muted-foreground">
              {evalResults
                ? <>Real data · <span className="font-mono text-neon">{evalResults.source_file}</span> · {evalResults.summary.total} questions</>
                : "Awaiting evaluation run — click Run Evaluation above"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {loadingResults && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
            <button
              onClick={fetchResults}
              className="text-[10px] text-neon hover:underline uppercase tracking-wider"
              title="Refresh chart from disk"
            >
              Refresh
            </button>
            {evalResults ? (
              <CheckCircle2 className="h-4 w-4 text-success" />
            ) : (
              <Database className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </div>

        {/* Summary stats row */}
        {evalResults && (
          <div className="mb-4 grid grid-cols-3 gap-3">
            <div className="rounded-xl border border-border/60 bg-background/30 p-3 text-center">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Total Evaluated</div>
              <div className="mt-1 text-2xl font-bold tabular-nums text-neon">{evalResults.summary.total}</div>
            </div>
            <div className="rounded-xl border border-success/30 bg-success/5 p-3 text-center">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">High Conf (≥0.80)</div>
              <div className="mt-1 text-2xl font-bold tabular-nums text-success">{evalResults.summary.high_conf}</div>
            </div>
            <div className="rounded-xl border border-warning/30 bg-warning/5 p-3 text-center">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Low Conf Caught</div>
              <div className="mt-1 text-2xl font-bold tabular-nums text-warning">{evalResults.summary.low_conf}</div>
            </div>
          </div>
        )}

        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            {evalResults && evalResults.bins.length > 0 ? (
              <BarChart
                data={evalResults.bins.map(b => ({ score: b.score.toFixed(2), count: b.count }))}
                margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="bargrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.85 0.18 195)" stopOpacity={0.9} />
                    <stop offset="100%" stopColor="oklch(0.85 0.18 195)" stopOpacity={0.3} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="oklch(0.30 0.03 235 / 50%)" />
                <XAxis
                  dataKey="score"
                  tick={{ fill: "oklch(0.68 0.025 230)", fontSize: 10 }}
                  label={{ value: "Confidence Score", position: "insideBottom", offset: -2, fill: "oklch(0.68 0.025 230)", fontSize: 10 }}
                />
                <YAxis tick={{ fill: "oklch(0.68 0.025 230)", fontSize: 11 }} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "oklch(0.21 0.028 240)",
                    border: "1px solid oklch(0.32 0.03 235)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => [v, "Questions"]}
                  labelFormatter={(l) => `Score bucket: ${l}`}
                />
                {/* Reference lines at decision thresholds */}
                <ReferenceLine x="0.55" stroke="oklch(0.75 0.15 50)" strokeDasharray="4 3"
                  label={{ value: "LOW", position: "top", fill: "oklch(0.75 0.15 50)", fontSize: 9 }} />
                <ReferenceLine x="0.80" stroke="oklch(0.72 0.18 150)" strokeDasharray="4 3"
                  label={{ value: "HIGH", position: "top", fill: "oklch(0.72 0.18 150)", fontSize: 9 }} />
                <Bar dataKey="count" fill="url(#bargrad)" radius={[3, 3, 0, 0]} />
              </BarChart>
            ) : (
              /* Placeholder while no data */
              <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
                <Database className="h-8 w-8 opacity-30" />
                <p className="text-xs">
                  {loadingResults ? "Loading results…" : "No evaluation results found. Run an evaluation first."}
                </p>
              </div>
            )}
          </ResponsiveContainer>
        </div>

        {evalResults?.summary.mean != null && (
          <div className="mt-3 flex items-center gap-2 text-[11px] text-muted-foreground">
            <TrendingUp className="h-3.5 w-3.5 text-neon" />
            Mean confidence score: <span className="font-mono font-semibold text-neon">{evalResults.summary.mean.toFixed(3)}</span>
            {evalResults.summary.mean < 0.55 && (
              <span className="ml-2 flex items-center gap-1 text-warning">
                <AlertTriangle className="h-3 w-3" /> Majority of questions are low-confidence — corpus may need expansion
              </span>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function ProgRow({
  label,
  done,
  total,
  accent,
  done_icon,
}: {
  label: string;
  done: number;
  total: number;
  accent?: boolean;
  done_icon?: string;
}) {
  const pct = total === 0 ? 0 : Math.round((done / total) * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className={cn("font-medium", accent && "text-neon")}>{label}</span>
        <span className="font-mono tabular-nums text-muted-foreground">
          {done} / {total} ({pct}%) {done_icon}
        </span>
      </div>
      <Progress value={pct} className="h-1.5" />
    </div>
  );
}

function MiniStat({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-background/30 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className={cn("mt-1 text-2xl font-bold tabular-nums", color)}>{value}</div>
    </div>
  );
}
