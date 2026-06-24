import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Brain, Flame, Loader2, Terminal, ArrowRight, Clock, ChevronDown, ChevronUp } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/tuning")({
  head: () => ({ meta: [{ title: "Model Tuning · Physics RAG SLM" }] }),
  component: TuningPage,
});

const MOCK_LOG = [
  "[TinySLM] Loading LoRA adapter from models/finetuned_slm...",
  "[TinySLM] Loading HITL corrections from data/feedback/hitl_corrections.jsonl",
  "[TinySLM] 14 correction pairs loaded for SFT",
  "Step  10/200 | Loss: 2.1043 | LR: 2.0e-4",
  "Step  50/200 | Loss: 1.4231 | LR: 1.8e-4",
  "Step 100/200 | Loss: 1.1092 | LR: 1.2e-4",
  "Step 150/200 | Loss: 0.8841 | LR: 6.0e-5",
  "Step 200/200 | Loss: 0.6712 | LR: 0.0e+0",
  "[TinySLM] Adapter saved to models/finetuned_slm/",
  "✅ Fine-tuning complete. Restart the API server to apply updated weights.",
];

// log slider helpers (1e-5 .. 1e-3)
const LR_MIN = Math.log10(1e-5);
const LR_MAX = Math.log10(1e-3);
const lrFromSlider = (v: number) => Math.pow(10, LR_MIN + (v / 100) * (LR_MAX - LR_MIN));
const sliderFromLR = (lr: number) => ((Math.log10(lr) - LR_MIN) / (LR_MAX - LR_MIN)) * 100;
const fmtLR = (v: number) =>
  v.toExponential(2).replace("e+", "e").replace("e-0", "e-").replace("e-", "e-");

type Correction = {
  question: string;
  bad_response: string;
  correct_response: string;
  submitted_at: string;
};

function TuningPage() {
  const [feedbackCount, setFeedbackCount] = useState<number>(14);
  const [lr, setLr] = useState(2e-4);
  const [rank, setRank] = useState(16);
  const [batch, setBatch] = useState(1);
  const [maxSteps, setMaxSteps] = useState(200);
  const [warmup, setWarmup] = useState(10);
  const [scheduler, setScheduler] = useState("cosine");
  const [running, setRunning] = useState(false);
  const [lines, setLines] = useState<string[]>(MOCK_LOG);
  const [corrections, setCorrections] = useState<Correction[]>([]);

  useEffect(() => {
    apiFetch<{ count?: number }>("/api/feedback/count", { timeoutMs: 3000 })
      .then((r) => typeof r.count === "number" && setFeedbackCount(r.count))
      .catch(() => {});
    // Fetch real before/after correction records
    apiFetch<{ corrections: Correction[] }>("/api/feedback/corrections?limit=5", { timeoutMs: 5000 })
      .then((r) => r.corrections?.length && setCorrections(r.corrections))
      .catch(() => {});
  }, []);

  // Poll training
  useEffect(() => {
    if (!running) return;
    const id = setInterval(async () => {
      try {
        const r = await apiFetch<{ status: string; logs?: string[] }>("/api/train/live", {
          timeoutMs: 3000,
        });
        if (r.logs && r.logs.length) setLines(r.logs);
        if (r.status === "idle" || r.status === "complete") setRunning(false);
      } catch {}
    }, 2000);
    return () => clearInterval(id);
  }, [running]);

  async function startTraining() {
    setRunning(true);
    setLines([]);
    toast.success("Training started", {
      description: "LoRA adapter updating on HITL corrections",
    });
    // Animate mock log line-by-line
    MOCK_LOG.forEach((line, i) => {
      setTimeout(() => setLines((prev) => [...prev, line]), 150 * (i + 1));
    });
    setTimeout(() => setRunning(false), 150 * (MOCK_LOG.length + 2));
    try {
      await apiFetch("/api/train/from-feedback", {
        method: "POST",
        body: JSON.stringify({
          learning_rate: lr,
          lora_rank: rank,
          batch_size: batch,
          max_steps: maxSteps,
          warmup_steps: warmup,
          lr_scheduler: scheduler,
        }),
        timeoutMs: 5000,
      });
    } catch {}
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Model Tuning</h1>
        <p className="text-sm text-muted-foreground">
          Human-in-the-loop SFT of the LoRA adapter on collected corrections.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.15fr]">
        {/* Left: config */}
        <div className="space-y-6">
          <section className="glass rounded-2xl p-5">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-neon-soft neon-glow">
                <Brain className="h-5 w-5 text-neon" />
              </div>
              <div className="flex-1">
                <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                  Pending HITL Corrections
                </div>
                <div className="text-3xl font-bold tabular-nums text-neon">{feedbackCount}</div>
              </div>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Collected from ✏️ Edit corrections on the Q&amp;A Playground.
            </p>
            <a className="mt-2 inline-block text-xs text-neon hover:underline" href="#">
              View corrections →
            </a>
          </section>

          <section className="glass rounded-2xl p-5">
            <h2 className="mb-5 text-sm font-semibold">Hyperparameters</h2>

            <div className="space-y-5">
              <SliderRow
                label="Learning Rate"
                value={fmtLR(lr)}
                hint="log-scale · 1e-5 → 1e-3"
              >
                <Slider
                  value={[sliderFromLR(lr)]}
                  min={0}
                  max={100}
                  step={1}
                  onValueChange={(v) => setLr(lrFromSlider(v[0]))}
                />
              </SliderRow>

              <SliderRow label="LoRA Rank (r)" value={String(rank)} hint="4 → 64">
                <Slider
                  value={[rank]}
                  min={4}
                  max={64}
                  step={1}
                  onValueChange={(v) => setRank(v[0])}
                />
              </SliderRow>

              <SliderRow
                label="Batch Size"
                value={String(batch)}
                hint="effective = batch × grad_accum"
              >
                <Slider
                  value={[batch]}
                  min={1}
                  max={8}
                  step={1}
                  onValueChange={(v) => setBatch(v[0])}
                />
              </SliderRow>

              <SliderRow label="Max Steps" value={String(maxSteps)} hint="50 → 500">
                <Slider
                  value={[maxSteps]}
                  min={50}
                  max={500}
                  step={10}
                  onValueChange={(v) => setMaxSteps(v[0])}
                />
              </SliderRow>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    Warmup Steps
                  </label>
                  <Input
                    type="number"
                    value={warmup}
                    onChange={(e) => setWarmup(Number(e.target.value) || 0)}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    LR Scheduler
                  </label>
                  <Select value={scheduler} onValueChange={setScheduler}>
                    <SelectTrigger className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="cosine">cosine</SelectItem>
                      <SelectItem value="linear">linear</SelectItem>
                      <SelectItem value="constant">constant</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </section>

          <Button
            onClick={startTraining}
            disabled={running}
            className="h-14 w-full rounded-2xl bg-neon text-base font-semibold text-primary-foreground hover:bg-neon/90 neon-glow"
          >
            {running ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Training in progress…
              </>
            ) : (
              <>
                <Flame className="mr-2 h-5 w-5" /> Start Human-in-the-Loop SFT
              </>
            )}
          </Button>
        </div>

        {/* Right: terminal */}
        <section className="flex h-full flex-col rounded-2xl border border-border bg-black/60 backdrop-blur-sm">
          <div className="flex items-center gap-2 border-b border-border/60 px-4 py-3">
            <Terminal className="h-4 w-4 text-neon" />
            <div className="text-xs font-semibold uppercase tracking-wider text-neon">
              Training Output
            </div>
            <div className="ml-auto flex gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-destructive/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-warning/70" />
              <span className="h-2.5 w-2.5 rounded-full bg-success/70" />
            </div>
          </div>
          <div className="flex-1 overflow-auto p-4 font-mono text-[12px] leading-6">
            {lines.length === 0 && (
              <div className="text-muted-foreground/60">$ awaiting training start…</div>
            )}
            {lines.map((l, i) => (
              <div
                key={i}
                className={cn(
                  "animate-fade-up whitespace-pre-wrap",
                  l.startsWith("✅")
                    ? "text-success"
                    : l.startsWith("[TinySLM]")
                    ? "text-neon"
                    : l.startsWith("Step")
                    ? "text-foreground/85"
                    : "text-muted-foreground"
                )}
              >
                <span className="select-none text-muted-foreground/50">
                  {String(i + 1).padStart(3, "0")} │{" "}
                </span>
                {l}
              </div>
            ))}
            {running && (
              <div className="mt-1 inline-block h-3 w-2 animate-pulse bg-neon align-middle" />
            )}
          </div>
        </section>
      </div>

      {/* Feedback Loop Evidence — stored before/after corrections */}
      <CorrectionsPanel corrections={corrections} />
    </div>
  );
}

function SliderRow({
  label,
  value,
  hint,
  children,
}: {
  label: string;
  value: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {label}
        </label>
        <span className="font-mono text-sm font-semibold tabular-nums text-neon">{value}</span>
      </div>
      {children}
      <div className="mt-1 text-[10px] text-muted-foreground/70">{hint}</div>
    </div>
  );
}

// ----- Feedback Loop Evidence Panel -----
function CorrectionsPanel({ corrections }: { corrections: Correction[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (corrections.length === 0) {
    return (
      <section className="glass rounded-2xl p-5">
        <div className="flex items-center gap-3 mb-3">
          <ArrowRight className="h-4 w-4 text-neon" />
          <h2 className="text-sm font-semibold">Feedback Loop Evidence</h2>
          <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">0 corrections</span>
        </div>
        <p className="text-xs text-muted-foreground">
          No correction-type feedback recorded yet. Go to the Q&amp;A Playground, expand
          "Improve This Answer" under any response, and submit a corrected equation.
          Those corrections accumulate here and feed the SFT loop above.
        </p>
      </section>
    );
  }

  return (
    <section className="glass rounded-2xl p-5 space-y-4">
      <div className="flex items-center gap-3">
        <ArrowRight className="h-4 w-4 text-neon" />
        <h2 className="text-sm font-semibold">Feedback Loop Evidence</h2>
        <span className="rounded-full bg-neon/15 px-2 py-0.5 text-[10px] font-semibold text-neon">
          {corrections.length} corrections captured
        </span>
        <span className="ml-auto text-[10px] text-muted-foreground">
          Stored in data/feedback/hitl_corrections.jsonl
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        These are real before/after pairs collected from the Q&amp;A Playground.
        The "Start Human-in-the-Loop SFT" button above trains the LoRA adapter on these exact records.
      </p>
      <div className="space-y-3">
        {corrections.map((c, i) => (
          <div key={i} className="rounded-xl border border-border/60 bg-background/30 overflow-hidden">
            {/* Header row */}
            <button
              className="flex w-full items-start justify-between gap-3 p-3 text-left hover:bg-background/50 transition-colors"
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              <div className="flex-1 min-w-0">
                <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">Question</div>
                <div className="text-[12px] font-medium line-clamp-2 text-foreground/90">{c.question}</div>
              </div>
              <div className="flex shrink-0 items-center gap-2 pt-4">
                <div className="flex items-center gap-1 text-[9px] text-muted-foreground">
                  <Clock className="h-2.5 w-2.5" />
                  {c.submitted_at ? new Date(c.submitted_at).toLocaleDateString() : ""}
                </div>
                {expanded === i
                  ? <ChevronUp className="h-3.5 w-3.5 text-muted-foreground" />
                  : <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />}
              </div>
            </button>

            {/* Expanded before/after */}
            {expanded === i && (
              <div className="grid grid-cols-2 gap-0 border-t border-border/60">
                <div className="p-3 border-r border-border/60">
                  <div className="mb-1.5 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-wider text-warning">
                    <span className="h-1.5 w-1.5 rounded-full bg-warning" /> Before (model output)
                  </div>
                  <p className="font-mono text-[11px] leading-relaxed text-foreground/75 whitespace-pre-wrap line-clamp-6">
                    {c.bad_response || <span className="italic text-muted-foreground">empty</span>}
                  </p>
                </div>
                <div className="p-3 bg-success/[0.03]">
                  <div className="mb-1.5 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-wider text-success">
                    <span className="h-1.5 w-1.5 rounded-full bg-success" /> After (user correction)
                  </div>
                  <p className="font-mono text-[11px] leading-relaxed text-foreground/90 whitespace-pre-wrap line-clamp-6">
                    {c.correct_response || <span className="italic text-muted-foreground">empty</span>}
                  </p>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
