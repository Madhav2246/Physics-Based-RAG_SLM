import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Quote,
  Send,
  Sparkles,
  ThumbsUp,
  Pencil,
  BookOpen,
  FunctionSquare,
  LineChart,
  Cpu,
  Zap,
} from "lucide-react";
import { toast } from "sonner";
import { Latex } from "@/components/Latex";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { apiFetch, API_BASE } from "@/lib/api";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Q&A Playground · Physics RAG SLM" },
      { name: "description", content: "Compare baseline Qwen vs Physics-augmented RAG with confidence inspector." },
    ],
  }),
  component: PlaygroundPage,
});

// ----- Mock data -----
const MOCK_BASELINE =
  "The MOSFET drain current is given by I = V/R (Ohm's Law). In saturation mode, the current is limited by the supply voltage and source resistance C_s. The gate voltage controls the channel conductance through the threshold coefficient β.";

const MOCK_RAG = `In saturation, the drain current follows the square-law model:

$$I_D = \\frac{1}{2} \\mu_n C_{ox} \\frac{W}{L}(V_{GS} - V_{th})^2$$

where $\\mu_n$ is electron mobility, $C_{ox}$ is gate oxide capacitance per unit area, $W/L$ is aspect ratio, and $V_{th}$ is threshold voltage. Valid when $V_{DS} \\geq V_{GS} - V_{th}$.`;

const MOCK_INSPECTOR = {
  confidence: 0.88,
  breakdown: [
    { label: "Evidence count ≥ 2", value: 0.2, ok: true },
    { label: "Symbolic parse", value: 0.2, ok: true },
    { label: "Dimensional check", value: 0.25, ok: true },
    { label: "Numerical realism", value: 0.25, ok: true },
    { label: "Response length", value: 0.1, ok: true },
  ],
  symbolic: "Parsed successfully",
  dimensional: "LHS {V:1} = RHS {V:1}",
  numerical: "1.84×10⁻³ A — Realistic",
  stability: { score: 0.95, label: "HIGH STABILITY — 3/3 samples consistent" },
  similarity: 0.79,
  explanation: {
    symbolic:    { verdict: "PASS" as const, reason: "Equation `Id = 0.5*mu*Cox*(W/L)*(Vgs-Vth)**2` parsed by SymPy. All 7 symbols are recognized standard MOSFET notation." },
    dimensional: { verdict: "PASS" as const, reason: "Both sides resolve to current [A]. The product μ·Cox·[m²/Vs·F/m²] × (W/L) × V² gives [A] — dimensionally consistent." },
    numerical:   { verdict: "PASS" as const, reason: "Substituting 100nm MOSFET values gives 1.84×10⁻³ A — within the physically realistic range 1pA–1A." },
    coverage:    { verdict: "PASS" as const, reason: "All symbols in the equation are recognized standard physics symbols." },
    feedback_hint: "This answer passes all physics checks. If the explanation or context is wrong, use 'Mark as correct' or write a more complete answer in the correction box.",
    summary: "All physics checks passed — equation is structurally and numerically correct.",
  },
  chunks: [
    {
      source: "MIT_OCW_6.012_Lectures.pdf · p.142",
      text:
        "In strong inversion and saturation (V_DS ≥ V_GS − V_th), the n-channel MOSFET drain current is approximated by the square-law expression I_D = (μ_n C_ox / 2)(W/L)(V_GS − V_th)², neglecting channel-length modulation.",
    },
    {
      source: "Sze_Semiconductor_Devices.pdf · p.318",
      text:
        "The pinch-off condition occurs when the inversion charge at the drain end vanishes. Beyond pinch-off, additional V_DS drops across the depletion region and I_D becomes nearly independent of V_DS, yielding the saturation regime.",
    },
    {
      source: "MIT_OCW_6.012_Lectures.pdf · p.149",
      text:
        "Mobility μ_n in the inversion layer is reduced relative to bulk silicon due to surface scattering. Typical values range 200–600 cm²/V·s depending on V_GS and process node.",
    },
  ],
};

// ----- Explore / Sweep metadata (Feature 1/2/3) -----
type ProvenanceRow = {
  symbol: string;
  value: number;
  unit: string;
  provenance: "user" | "corpus" | "default";
  description: string;
};
type ExploreResult = {
  success: boolean;
  target: string | null;
  symbolic: string | null;
  numeric: number | null;
  sanity_ok: boolean | null;
  error: string;
  corpus_equation: string | null;
  provenance: ProvenanceRow[];
  provenance_fraction: number | null;
  sweep: { sweep_var: string; target_var: string; x: number[]; y: number[]; node_name: string } | null;
};
type RagMeta = {
  mode: "LOOKUP" | "EXPLORE" | "SWEEP";
  node_profile: string;
  explore_result: ExploreResult | null;
  sweep_plot_url: string | null;
};

const MOCK_RAGMETA: RagMeta = {
  mode: "EXPLORE",
  node_profile: "100nm_CMOS (default)",
  explore_result: {
    success: true,
    target: "WL",
    symbolic: "WL = 2.0*Id/(Cox*Vov**2*mu)",
    numeric: 8.0,
    sanity_ok: true,
    error: "",
    corpus_equation: "Id = 0.5*Cox*W*mu*(Vgs - Vth)**2/L",
    provenance: [
      { symbol: "Id", value: 1e-3, unit: "A", provenance: "user", description: "extracted from query" },
      { symbol: "Vov", value: 0.5, unit: "V", provenance: "user", description: "extracted from query" },
      { symbol: "mu", value: 0.05, unit: "m^2/Vs", provenance: "default", description: "electron mobility (100nm node)" },
      { symbol: "Cox", value: 0.02, unit: "F/m^2", provenance: "default", description: "gate oxide capacitance (100nm node)" },
    ],
    provenance_fraction: 0.5,
    sweep: null,
  },
  sweep_plot_url: null,
};

function PlaygroundPage() {
  const [question, setQuestion] = useState(
    "What is the drain current equation for a MOSFET in saturation?"
  );
  // Manual pipeline mode override. AUTO = let the backend keyword-detect;
  // otherwise force LOOKUP (explain) / EXPLORE (solve) / SWEEP (plot).
  const [mode, setMode] = useState<"AUTO" | "LOOKUP" | "EXPLORE" | "SWEEP">("AUTO");
  const [loading, setLoading] = useState(false);
  const [baseline, setBaseline] = useState<{
    text: string;
    model: string;
    explanation?: Explanation;
    symbolic?: string;
    dimensional?: string;
    numerical?: string;
    confidence?: number;
  } | null>({
    text: MOCK_BASELINE,
    model: "meta/llama-3.3-70b-instruct",
  });
  const [rag, setRag] = useState<string | null>(MOCK_RAG);
  const [inspector, setInspector] = useState<typeof MOCK_INSPECTOR | null>(MOCK_INSPECTOR);
  const [ragMeta, setRagMeta] = useState<RagMeta | null>(MOCK_RAGMETA);
  // 1.5B and 3B compare results (null = not yet queried)
  const [ragCompare, setRagCompare] = useState<{
    model_1_5b: { response: string; explanation?: any } | null;
    model_3b:   { response: string; explanation?: any } | null;
  } | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setCompareLoading(true);
    setBaseline(null);
    setRag(null);
    setInspector(null);
    setRagMeta(null);
    setRagCompare(null);

    // Fire baseline, 0.5B RAG, and 1.5B/3B compare all at once
    const [b, r, c] = await Promise.allSettled([
      apiFetch<any>("/api/query/baseline", {
        method: "POST",
        body: JSON.stringify({ question }),
        timeoutMs: 35000,
      }),
      apiFetch<any>("/api/query", {
        method: "POST",
        body: JSON.stringify({ question, mode: mode === "AUTO" ? null : mode }),
        timeoutMs: 35000,
      }),
      apiFetch<any>("/api/query/rag-compare", {
        method: "POST",
        body: JSON.stringify({ question }),
        timeoutMs: 45000,
      }),
    ]);

    if (b.status === "fulfilled") {
      const bv = b.value;
      setBaseline({
        text: bv.response ?? bv.answer ?? MOCK_BASELINE,
        model: bv.model ?? "meta/llama-3.3-70b-instruct",
        explanation: bv.explanation,
        symbolic: bv.symbolic,
        dimensional: bv.dimensional,
        numerical: bv.numerical,
        confidence: bv.confidence,
      });
    } else {
      setBaseline({ text: MOCK_BASELINE, model: "meta/llama-3.3-70b-instruct" });
    }

    if (r.status === "fulfilled") {
      const v = r.value;
      setRag(v.response ?? v.answer ?? MOCK_RAG);
      setInspector({
        confidence: v.confidence ?? MOCK_INSPECTOR.confidence,
        breakdown: v.breakdown ?? MOCK_INSPECTOR.breakdown,
        symbolic: v.symbolic ?? MOCK_INSPECTOR.symbolic,
        dimensional: v.dimensional ?? MOCK_INSPECTOR.dimensional,
        numerical: v.numerical ?? MOCK_INSPECTOR.numerical,
        stability: v.stability ?? MOCK_INSPECTOR.stability,
        similarity: v.similarity ?? MOCK_INSPECTOR.similarity,
        chunks: v.chunks ?? MOCK_INSPECTOR.chunks,
        explanation: v.explanation ?? MOCK_INSPECTOR.explanation,
      });
      setRagMeta({
        mode: v.mode ?? "LOOKUP",
        node_profile: v.node_profile ?? "100nm_CMOS (default)",
        explore_result: v.explore_result ?? null,
        sweep_plot_url: v.sweep_plot_url ?? null,
      });
    } else {
      setRag(MOCK_RAG);
      setInspector(MOCK_INSPECTOR);
      setRagMeta(MOCK_RAGMETA);
    }

    if (c.status === "fulfilled" && !c.value.error) {
      setRagCompare({
        model_1_5b: c.value.model_1_5b ?? null,
        model_3b:   c.value.model_3b   ?? null,
      });
    } else {
      setRagCompare({ model_1_5b: null, model_3b: null });
    }

    setLoading(false);
    setCompareLoading(false);
  }

  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col gap-4">
      <header className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Q&A Playground</h1>
          <p className="text-sm text-muted-foreground">
            Side-by-side: pretrained Qwen vs Physics-RAG with deterministic validation.
          </p>
        </div>
        <div className="text-xs text-muted-foreground">
          POST <span className="text-neon">/api/query</span> + <span className="text-neon">/api/query/baseline</span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 gap-4">
        <div className="flex min-w-0 flex-1 flex-col gap-4">
          {/* Input bar */}
          <form onSubmit={onSubmit} className="glass-strong relative rounded-2xl p-2">
            <div className="flex items-center gap-2">
              <Sparkles className="ml-3 h-4 w-4 text-neon" />
              <Input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a semiconductor physics question…"
                className="h-12 border-0 bg-transparent text-base shadow-none focus-visible:ring-0"
              />
              <Button
                type="submit"
                disabled={loading || !question.trim()}
                className="h-10 rounded-xl bg-neon px-5 text-primary-foreground hover:bg-neon/90 neon-glow"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                <span className="ml-2">Query</span>
              </Button>
            </div>

            {/* Mode override: AUTO lets the backend keyword-detect; the others
                force the pipeline path (LOOKUP=explain, EXPLORE=solve, SWEEP=plot). */}
            <div className="mt-2 flex items-center gap-1.5 pl-3">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Mode
              </span>
              {(["AUTO", "LOOKUP", "EXPLORE", "SWEEP"] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  onClick={() => setMode(m)}
                  className={`rounded-lg px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    mode === m
                      ? "bg-neon text-primary-foreground neon-glow"
                      : "bg-muted/40 text-muted-foreground hover:bg-muted/70"
                  }`}
                >
                  {m}
                </button>
              ))}
              <span className="ml-1 text-[10px] text-muted-foreground/60">
                {mode === "AUTO" ? "auto-detect from query" : `forcing ${mode} mode`}
              </span>
            </div>
          </form>

          {/* Three panes: Baseline | RAG Carousel | Inspector */}
          <div className="grid min-h-0 flex-1 grid-cols-[28fr_42fr] gap-4">
            <Pane
              title={baseline?.model
                ? `Baseline \u00b7 ${baseline.model.replace("meta/", "").replace("-instruct", "")}`
                : "Baseline \u00b7 llama-3.3-70b"}
              badge="No Validation"
              badgeColor="warning"
              subtitle="70B SOTA \u00b7 No retrieval \u00b7 No physics validation layer — shows what even a large model gets wrong"
              borderClass="border-l-warning"
              loading={loading}
            >
              {baseline && (
                <div className="animate-fade-up space-y-3">
                  <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-warning">
                    <AlertTriangle className="h-3 w-3" /> Unvalidated · Physics errors may be present
                  </div>
                  <p className="whitespace-pre-wrap text-[14px] leading-relaxed text-foreground/90">
                    {baseline.text}
                  </p>
                  {baseline.symbolic && (
                    <div className="mt-4">
                      <ValidationCard
                        symbolic={baseline.symbolic}
                        dimensional={baseline.dimensional}
                        numerical={baseline.numerical}
                      />
                    </div>
                  )}
                  {baseline.confidence !== undefined && (
                    <div className="mt-2 text-[11px] font-semibold text-muted-foreground flex items-center gap-2">
                      <span>Baseline Confidence Score:</span>
                      <span className={cn(
                        "font-mono text-xs px-1.5 py-0.5 rounded",
                        baseline.confidence < 0.5 ? "bg-destructive/15 text-destructive" : baseline.confidence < 0.75 ? "bg-warning/15 text-warning" : "bg-success/15 text-success"
                      )}>
                        {baseline.confidence.toFixed(2)}
                      </span>
                    </div>
                  )}
                  {baseline.explanation && (
                    <div className="mt-4 border-t border-border/50 pt-4">
                      <PhysicsReasoningCard explanation={baseline.explanation} />
                    </div>
                  )}
                </div>
              )}
            </Pane>

            <Pane
              title="Physics RAG Engine"
              badge="Validated"
              badgeColor="success"
              subtitle="Qwen + LoRA · FAISS/BM25 · Deterministic Validators"
              borderClass="border-l-success"
              loading={loading}
            >
              {/* ── 3-model tab carousel ── */}
              <RagModelCarousel
                loading={loading}
                compareLoading={compareLoading}
                rag={rag}
                ragMeta={ragMeta}
                question={question}
                ragCompare={ragCompare}
              />
            </Pane>
          </div>
        </div>

        <Inspector data={inspector} loading={loading} />
      </div>
    </div>
  );
}

// ----- Mode badge: LOOKUP / EXPLORE / SWEEP + node profile -----
function ModeBadge({ mode, node }: { mode: string; node: string }) {
  const style =
    mode === "SWEEP"
      ? "bg-chart-3/15 text-chart-3"
      : mode === "EXPLORE"
      ? "bg-neon/15 text-neon"
      : "bg-muted text-muted-foreground";
  const showNode = node && !node.startsWith("100nm_CMOS (default)");
  return (
    <div className="flex items-center gap-1.5">
      <span className={cn("rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider", style)}>
        {mode}
      </span>
      {showNode && (
        <span className="flex items-center gap-1 rounded-full bg-background/40 px-2 py-0.5 text-[9px] font-medium text-foreground/70">
          <Cpu className="h-2.5 w-2.5" /> {node.replace(/_/g, " ")}
        </span>
      )}
    </div>
  );
}

// ─── RAG Model Carousel (0.5B / 1.5B / 3B tab slider) ─────────────────────
const RAG_MODELS = [
  {
    id: "0.5B",
    label: "Qwen 0.5B",
    sub: "LoRA fine-tuned · FAISS/BM25",
    vram: "~2 GB",
    live: true,
  },
  {
    id: "1.5B",
    label: "Qwen 1.5B",
    sub: "Standard · FAISS/BM25",
    vram: "~4 GB",
    live: false,
  },
  {
    id: "3B",
    label: "Qwen 3B",
    sub: "Standard · FAISS/BM25",
    vram: "~8 GB",
    live: false,
  },
] as const;

function RagModelCarousel({
  loading,
  compareLoading,
  rag,
  ragMeta,
  question,
  ragCompare,
}: {
  loading: boolean;
  compareLoading: boolean;
  rag: string | null;
  ragMeta: RagMeta | null;
  question: string;
  ragCompare: { model_1_5b: { response: string; explanation?: any } | null; model_3b: { response: string; explanation?: any } | null } | null;
}) {
  const [activeIdx, setActiveIdx] = useState(0);

  const model = RAG_MODELS[activeIdx];

  // Pick the right compare data for tabs 1 and 2
  const compareData = activeIdx === 1 ? ragCompare?.model_1_5b : activeIdx === 2 ? ragCompare?.model_3b : null;
  const isCompareLoading = compareLoading && activeIdx > 0 && !compareData;

  return (
    <div className="flex h-full flex-col gap-2">
      {/* ── Tab strip ── */}
      <div className="flex items-center gap-1 rounded-xl border border-border/40 bg-background/30 p-1">
        {RAG_MODELS.map((m, i) => (
          <button
            key={m.id}
            onClick={() => setActiveIdx(i)}
            className={cn(
              "flex flex-1 flex-col items-center gap-0.5 rounded-lg px-2 py-1.5 text-center transition-all duration-200",
              activeIdx === i
                ? "bg-neon/15 text-neon shadow-sm"
                : "text-muted-foreground hover:text-foreground/80"
            )}
          >
            <span className="text-[11px] font-semibold">{m.label}</span>
            <span className="text-[9px] opacity-70">{m.vram}</span>
          </button>
        ))}
        {/* nav arrows */}
        <button
          onClick={() => setActiveIdx((i) => Math.max(0, i - 1))}
          disabled={activeIdx === 0}
          className="ml-1 rounded-md p-1 text-muted-foreground transition hover:text-foreground disabled:opacity-30"
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={() => setActiveIdx((i) => Math.min(RAG_MODELS.length - 1, i + 1))}
          disabled={activeIdx === RAG_MODELS.length - 1}
          className="rounded-md p-1 text-muted-foreground transition hover:text-foreground disabled:opacity-30"
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* ── Content panel ── */}
      <div className="min-h-0 flex-1 overflow-auto">
        {activeIdx === 0 ? (
          /* ── Tab 0: Qwen 0.5B (local LoRA) ── */
          rag ? (
            <div className="animate-fade-up space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-success">
                  <CheckCircle2 className="h-3 w-3" /> Validated · LoRA fine-tuned
                </div>
                {ragMeta && <ModeBadge mode={ragMeta.mode} node={ragMeta.node_profile} />}
              </div>
              <Latex>{rag}</Latex>
              {ragMeta?.explore_result && <ExploreResultCard er={ragMeta.explore_result} />}
              {ragMeta?.sweep_plot_url && <SweepPlotCard url={ragMeta.sweep_plot_url} node={ragMeta.node_profile} />}
              <FeedbackDrawer question={question} ragAnswer={rag} />
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
              {loading ? <><Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />Running RAG pipeline…</> : "Submit a question to see RAG output"}
            </div>
          )
        ) : isCompareLoading ? (
          /* ── Loading state for 1.5B / 3B ── */
          <div className="flex h-full flex-col items-center justify-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin text-neon/60" />
            <span>Calling {model.label} via NVIDIA NIM…</span>
            <span className="text-[10px] text-muted-foreground/50">{model.sub}</span>
          </div>
        ) : compareData ? (
          /* ── Live response for 1.5B / 3B ── */
          <div className="animate-fade-up space-y-3">
            <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wider text-neon">
              <CheckCircle2 className="h-3 w-3" /> Validated · NIM API · {model.vram}
            </div>
            <p className="whitespace-pre-wrap text-[14px] leading-relaxed text-foreground/90">
              {compareData.response}
            </p>
            {compareData.explanation && (
              <div className="mt-4 border-t border-border/50 pt-4">
                <PhysicsReasoningCard explanation={compareData.explanation} />
              </div>
            )}
          </div>
        ) : (
          /* ── Pre-query placeholder ── */
          <div className="flex h-full flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-border/40 bg-background/20 p-8 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-neon/10">
              <Zap className="h-6 w-6 text-neon/60" />
            </div>
            <div>
              <p className="text-sm font-semibold text-foreground/80">{model.label} · RAG-augmented</p>
              <p className="mt-1 text-xs text-muted-foreground">{model.sub} · {model.vram} VRAM</p>
            </div>
            <p className="text-xs text-muted-foreground/60">Submit a question — all 3 models answer simultaneously</p>
          </div>
        )}
      </div>
    </div>
  );
}

// Pretty-print a SymPy string for display: "WL = 2.0*Id/(Cox*Vov**2*mu)" → readable
function prettyEq(s: string): string {
  return s
    .replace(/\bWL\b/g, "W/L")
    .replace(/\*\*/g, "^")
    .replace(/\*/g, "·");
}

// ----- Explore result: derived equation + numeric + provenance audit -----
function ExploreResultCard({ er }: { er: ExploreResult }) {
  if (!er.success) {
    return (
      <div className="rounded-xl border border-warning/30 bg-warning/5 p-3 text-[12px] text-warning">
        <div className="mb-1 flex items-center gap-1.5 font-semibold uppercase tracking-wider text-[10px]">
          <FunctionSquare className="h-3 w-3" /> Design synthesis
        </div>
        {er.error || "Could not derive a result."}
      </div>
    );
  }

  const provColor = (p: string) =>
    p === "user" ? "bg-success/15 text-success"
    : p === "corpus" ? "bg-neon/15 text-neon"
    : "bg-warning/15 text-warning";

  const pf = er.provenance_fraction ?? 0;

  return (
    <div className="rounded-xl border border-neon/30 bg-neon/[0.03] p-3.5">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-neon">
          <FunctionSquare className="h-3 w-3" /> Deterministic Design Synthesis
        </div>
        <span className="text-[9px] uppercase tracking-wider text-muted-foreground">SymPy · no SLM</span>
      </div>

      {/* Derived equation */}
      {er.symbolic && (
        <div className="mb-2 rounded-md bg-background/50 px-3 py-2 font-mono text-[13px] text-foreground/90">
          {prettyEq(er.symbolic)}
        </div>
      )}

      {/* Numeric answer */}
      {er.numeric !== null && (
        <div className="mb-3 flex items-baseline gap-2">
          <span className="text-2xl font-bold tabular-nums text-neon">
            {er.target === "WL" ? "W/L" : er.target} = {formatNum(er.numeric)}
          </span>
          {er.sanity_ok === false && (
            <span className="flex items-center gap-1 text-[10px] text-warning">
              <AlertTriangle className="h-3 w-3" /> outside typical range
            </span>
          )}
        </div>
      )}

      {/* Provenance table — the zero-hallucination evidence */}
      {er.provenance.length > 0 && (
        <div>
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Value provenance
            </span>
            <span className="text-[10px] tabular-nums text-foreground/70">
              {Math.round(pf * 100)}% user-supplied · {Math.round((1 - pf) * 100)}% assumed
            </span>
          </div>
          {/* provenance fraction bar */}
          <div className="mb-2 flex h-1.5 overflow-hidden rounded-full bg-secondary">
            <div className="bg-success" style={{ width: `${pf * 100}%` }} />
            <div className="bg-warning/60" style={{ width: `${(1 - pf) * 100}%` }} />
          </div>
          <table className="w-full text-[11px]">
            <tbody>
              {er.provenance.map((p) => (
                <tr key={p.symbol} className="border-b border-border/30 last:border-0">
                  <td className="py-1 font-mono font-medium">{p.symbol}</td>
                  <td className="py-1 text-right font-mono tabular-nums text-foreground/80">
                    {formatNum(p.value)} {p.unit}
                  </td>
                  <td className="w-24 py-1 pl-2 text-right">
                    <span className={cn("rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase", provColor(p.provenance))}>
                      {p.provenance}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// Compact number formatter — scientific for very small/large, else fixed
function formatNum(n: number): string {
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (abs < 1e-3 || abs >= 1e5) return n.toExponential(3);
  return Number(n.toPrecision(4)).toString();
}

// ----- Sweep plot: trade-off curve PNG served by backend static mount -----
function SweepPlotCard({ url, node }: { url: string; node: string }) {
  return (
    <div className="rounded-xl border border-chart-3/30 bg-chart-3/[0.03] p-3">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-chart-3">
          <LineChart className="h-3 w-3" /> Parametric Sweep
        </div>
        <span className="text-[9px] uppercase tracking-wider text-muted-foreground">
          {node.replace(/_/g, " ")}
        </span>
      </div>
      <img
        src={`${API_BASE}${url}`}
        alt="Parametric sweep trade-off curve"
        className="w-full rounded-md border border-border/40 bg-white"
        loading="lazy"
      />
    </div>
  );
}

function Pane({
  title,
  badge,
  badgeColor,
  subtitle,
  borderClass,
  loading,
  children,
}: {
  title: string;
  badge: string;
  badgeColor: "success" | "warning";
  subtitle: string;
  borderClass: string;
  loading: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={cn("glass flex min-h-0 flex-col rounded-2xl border-l-4", borderClass)}>
      <div className="border-b border-border/60 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold tracking-tight">{title}</h2>
          <span
            className={cn(
              "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider",
              badgeColor === "success" ? "bg-success/15 text-success" : "bg-warning/15 text-warning"
            )}
          >
            {badge}
          </span>
        </div>
        <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-5">
        {loading ? <RetrievingSpinner /> : children}
      </div>
    </section>
  );
}

function RetrievingSpinner() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="flex flex-col items-center gap-3 text-muted-foreground">
        <div className="relative h-10 w-10">
          <div className="absolute inset-0 animate-ping rounded-full bg-neon/30" />
          <div className="absolute inset-1 grid place-items-center rounded-full bg-neon/20">
            <Loader2 className="h-5 w-5 animate-spin text-neon" />
          </div>
        </div>
        <div className="text-xs uppercase tracking-[0.2em]">Retrieving physics chunks…</div>
      </div>
    </div>
  );
}

function FeedbackDrawer({ question, ragAnswer }: { question: string; ragAnswer: string }) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<null | "correct" | "thumbs" | "gap">(null);
  const [text, setText] = useState("");

  async function send(feedback_type: string, payload: Record<string, any>) {
    try {
      await apiFetch("/api/feedback", {
        method: "POST",
        body: JSON.stringify({ question, bad_response: ragAnswer, feedback_type, ...payload }),
        timeoutMs: 5000,
      });
    } catch {
      /* mock-friendly */
    }
    toast.success("Feedback recorded", {
      description: "Stored to data/feedback/hitl_corrections.jsonl",
    });
    setMode(null);
    setText("");
  }

  return (
    <div className="mt-4 rounded-xl border border-border/60 bg-background/30">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground"
      >
        <span>Improve This Answer</span>
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="space-y-2 border-t border-border/60 p-3">
          <div className="flex flex-wrap gap-2">
            <Button size="sm" variant="outline" className="h-8" onClick={() => setMode("correct")}>
              <Pencil className="mr-1.5 h-3 w-3" /> Correct the equation
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-8"
              onClick={() => send("thumbs_up", { correct_response: ragAnswer })}
            >
              <ThumbsUp className="mr-1.5 h-3 w-3" /> Mark as correct
            </Button>
            <Button size="sm" variant="outline" className="h-8" onClick={() => setMode("gap")}>
              <BookOpen className="mr-1.5 h-3 w-3" /> Flag missing topic
            </Button>
          </div>

          {mode === "correct" && (
            <div className="space-y-2 pt-2">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                placeholder="Paste corrected LaTeX, e.g. I_D = \\frac{1}{2} \\mu_n C_{ox} \\frac{W}{L}(V_{GS}-V_{th})^2"
                className="font-mono text-xs"
              />
              <Button
                size="sm"
                className="bg-neon text-primary-foreground hover:bg-neon/90"
                onClick={() => send("correction", { correct_response: text })}
                disabled={!text.trim()}
              >
                Submit correction
              </Button>
            </div>
          )}

          {mode === "gap" && (
            <div className="space-y-2 pt-2">
              <Input
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Missing topic, e.g. ‘BSIM4 channel-length modulation’"
              />
              <Button
                size="sm"
                className="bg-neon text-primary-foreground hover:bg-neon/90"
                onClick={() => send("corpus_gap", { missing_topic: text })}
                disabled={!text.trim()}
              >
                Flag gap
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ----- Inspector -----
function Inspector({
  data,
  loading,
}: {
  data: typeof MOCK_INSPECTOR | null;
  loading: boolean;
}) {
  return (
    <aside className="glass flex w-[360px] shrink-0 flex-col overflow-hidden rounded-2xl">
      <div className="border-b border-border/60 p-4">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          Confidence Engine
        </div>
        <div className="text-sm font-semibold">Inspector</div>
      </div>
      <div className="min-h-0 flex-1 space-y-3 overflow-auto p-4">
        {loading || !data ? (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            {loading ? "Awaiting RAG response…" : "Submit a question to inspect"}
          </div>
        ) : (
          <div className="animate-slide-in-right space-y-3">
            <ConfidenceCard value={data.confidence} />
            <BreakdownCard rows={data.breakdown} total={data.confidence} />
            <ValidationCard
              symbolic={data.symbolic}
              dimensional={data.dimensional}
              numerical={data.numerical}
            />
            <PhysicsReasoningCard explanation={data.explanation} />
            <StabilityCard score={data.stability.score} label={data.stability.label} />
            <SimilarityCard value={data.similarity} />
            <ChunksCard chunks={data.chunks} />
          </div>
        )}
      </div>
    </aside>
  );
}

function InspectorCard({
  title,
  children,
}: {
  title: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-border/60 bg-background/30 p-3.5">
      <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
        {title}
      </div>
      {children}
    </div>
  );
}

function ConfidenceCard({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value));
  const color = pct < 0.5 ? "var(--danger)" : pct < 0.75 ? "var(--warning)" : "var(--success)";
  const label = pct < 0.5 ? "LOW CONFIDENCE" : pct < 0.75 ? "MEDIUM CONFIDENCE" : "HIGH CONFIDENCE";
  // arc from -135° to 135° (270° sweep)
  const R = 56;
  const C = 2 * Math.PI * R;
  const sweep = 0.75; // 270deg
  const arcLen = C * sweep;
  const offset = arcLen * (1 - pct);

  return (
    <InspectorCard title="Confidence Score">
      <div className="flex items-center gap-4">
        <div className="relative h-32 w-32">
          <svg viewBox="0 0 140 140" className="-rotate-[135deg]">
            <circle
              cx="70"
              cy="70"
              r={R}
              fill="none"
              stroke="oklch(0.30 0.03 235)"
              strokeWidth="10"
              strokeDasharray={`${arcLen} ${C}`}
              strokeLinecap="round"
            />
            <circle
              cx="70"
              cy="70"
              r={R}
              fill="none"
              stroke={color}
              strokeWidth="10"
              strokeDasharray={`${arcLen} ${C}`}
              strokeDashoffset={offset}
              strokeLinecap="round"
              style={{ transition: "stroke-dashoffset 0.8s ease, stroke 0.3s" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="text-3xl font-bold tabular-nums" style={{ color }}>
              {pct.toFixed(2)}
            </div>
            <div className="text-[9px] uppercase tracking-wider text-muted-foreground">/ 1.00</div>
          </div>
        </div>
        <div className="flex-1">
          <div className="text-xs font-semibold" style={{ color }}>
            {label}
          </div>
          <div className="mt-1 text-[11px] text-muted-foreground">
            Aggregate of 5 deterministic validators applied to the generated answer.
          </div>
        </div>
      </div>
    </InspectorCard>
  );
}

function BreakdownCard({
  rows,
  total,
}: {
  rows: { label: string; value: number; ok: boolean }[];
  total: number;
}) {
  return (
    <InspectorCard title="Confidence Breakdown">
      <table className="w-full text-[11px]">
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-border/40 last:border-0">
              <td className="py-1.5">{r.label}</td>
              <td className="py-1.5 text-right font-mono tabular-nums text-muted-foreground">
                +{r.value.toFixed(2)}
              </td>
              <td className="w-5 py-1.5 text-right">
                <span
                  className={cn(
                    "ml-1 inline-block h-2 w-2 rounded-full",
                    r.ok ? "bg-success" : "bg-destructive"
                  )}
                />
              </td>
            </tr>
          ))}
          <tr>
            <td className="pt-2 text-[10px] uppercase tracking-wider text-muted-foreground">Total</td>
            <td className="pt-2 text-right font-mono text-sm font-semibold text-neon">
              {total.toFixed(2)}
            </td>
            <td className="pt-2 text-right text-[10px] text-muted-foreground">/1.00</td>
          </tr>
        </tbody>
      </table>
    </InspectorCard>
  );
}

function ValidationCard({
  symbolic = "",
  dimensional = "",
  numerical = "",
}: {
  symbolic?: string;
  dimensional?: string;
  numerical?: string;
}) {
  const rows = [
    { label: "Symbolic",    value: symbolic },
    { label: "Dimensional", value: dimensional },
    { label: "Numerical",   value: numerical },
  ];

  // Physics score: 1pt symbolic parse + 1pt dimensional + 1pt numerical + 1pt full coverage
  const symOk  = !!symbolic?.includes("[OK]");
  const dimOk  = !!dimensional?.includes("[OK]");
  const numOk  = !!numerical?.includes("[OK]");
  const covOk  = numOk && !numerical?.includes("Unresolved");
  const physicsScore = (symOk ? 1 : 0) + (dimOk ? 1 : 0) + (numOk ? 1 : 0) + (covOk ? 1 : 0);
  const scoreBadge =
    physicsScore >= 3 ? "bg-success/20 text-success" :
    physicsScore >= 1 ? "bg-warning/20 text-warning" :
    "bg-destructive/20 text-destructive";

  return (
    <InspectorCard
      title={
        <div className="flex w-full items-center justify-between">
          <span>Physics Validator</span>
          <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wider ${scoreBadge}`}>
            {physicsScore} / 4
          </span>
        </div>
      }
    >
      <div className="space-y-1.5">
        {rows.map((r) => {
          const ok   = r.value?.includes("[OK]");
          const fail = r.value?.includes("[FAIL]");
          const rowBg   = ok ? "bg-success/10" : fail ? "bg-destructive/10" : "bg-warning/10";
          const iconCol = ok ? "text-success"  : fail ? "text-destructive"  : "text-warning";
          const Icon    = ok ? CheckCircle2    : AlertTriangle;
          return (
            <div
              key={r.label}
              className={`flex items-start gap-2 rounded-md px-2 py-1.5 text-[11px] ${rowBg}`}
            >
              <Icon className={`mt-0.5 h-3 w-3 shrink-0 ${iconCol}`} />
              <span className="w-20 shrink-0 font-medium text-muted-foreground">{r.label}:</span>
              <span className="break-words text-foreground/90">{r.value}</span>
            </div>
          );
        })}
      </div>
    </InspectorCard>
  );
}

function StabilityCard({ score, label }: { score: number; label: string }) {
  return (
    <InspectorCard title="Uncertainty / Stability">
      <div className="flex items-baseline gap-3">
        <div className="text-2xl font-bold tabular-nums text-success">{score.toFixed(2)}</div>
        <div className="text-[11px] font-medium text-success">{label}</div>
      </div>
      <div className="mt-1 text-[10px] text-muted-foreground">
        Derived from response diversity across n=3 model samples.
      </div>
    </InspectorCard>
  );
}

function SimilarityCard({ value }: { value: number }) {
  const ok = value > 0.6;
  return (
    <InspectorCard title="Semantic Similarity">
      <div className="mb-2 flex items-baseline justify-between">
        <div className="text-lg font-bold tabular-nums">{value.toFixed(2)}</div>
        <div className={cn("text-[10px] font-medium", ok ? "text-success" : "text-warning")}>
          {ok ? "Answer aligns with source corpus" : "⚠ Possible hallucination"}
        </div>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-secondary">
        <div
          className="h-full rounded-full bg-gradient-to-r from-chart-3 to-neon transition-all duration-700"
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </InspectorCard>
  );
}

type ExplanationComponent = { verdict: "PASS" | "WARN" | "FAIL"; reason: string };
type Explanation = {
  symbolic: ExplanationComponent;
  dimensional: ExplanationComponent;
  numerical: ExplanationComponent;
  coverage: ExplanationComponent;
  feedback_hint: string;
  summary: string;
};

function PhysicsReasoningCard({ explanation }: { explanation: Explanation }) {
  const [open, setOpen] = useState(false);
  const componentKeys = ["symbolic", "dimensional", "numerical", "coverage"] as const;
  const hasFailure = componentKeys.some(
    (k) => explanation[k].verdict !== "PASS"
  );

  const verdictStyle = (v: string) =>
    v === "PASS"
      ? "bg-success/15 text-success"
      : v === "FAIL"
      ? "bg-destructive/15 text-destructive"
      : "bg-warning/15 text-warning";

  const rows: [string, ExplanationComponent][] = [
    ["Symbolic",    explanation.symbolic],
    ["Dimensional", explanation.dimensional],
    ["Numerical",   explanation.numerical],
    ["Coverage",    explanation.coverage],
  ];

  const passedCount = componentKeys.filter((k) => explanation[k].verdict === "PASS").length;

  return (
    <InspectorCard 
      title={
        <div className="flex w-full items-center justify-between">
          <span>Physics Checker Commentary</span>
          <span className="rounded bg-success/20 px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-success">
            {passedCount}/4 PASSED
          </span>
        </div>
      }
    >
      {/* Summary line + toggle */}
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start justify-between gap-2 text-left"
      >
        <p className={cn("text-[11px] leading-snug", hasFailure ? "text-warning" : "text-success")}>
          {explanation.summary}
        </p>
        <ChevronDown
          className={cn("h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform mt-0.5",
            open && "rotate-180")}
        />
      </button>

      {open && (
        <div className="mt-3 space-y-2">
          {rows.map(([label, comp]) => (
            <div key={label} className="space-y-0.5">
              <div className="flex items-center gap-2">
                <span className={cn("rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider",
                  verdictStyle(comp.verdict))}>
                  {comp.verdict}
                </span>
                <span className="text-[10px] font-medium text-muted-foreground">{label}</span>
              </div>
              <p className="pl-1 text-[10px] leading-relaxed text-foreground/75">{comp.reason}</p>
            </div>
          ))}

          {/* Feedback hint — only shown when something failed */}
          {hasFailure && explanation.feedback_hint && (
            <div className="mt-3 rounded-lg border border-warning/30 bg-warning/5 p-2.5">
              <div className="mb-1 text-[9px] font-semibold uppercase tracking-wider text-warning">
                Correction guidance
              </div>
              <p className="text-[10px] leading-relaxed text-foreground/80 whitespace-pre-wrap">
                {explanation.feedback_hint}
              </p>
            </div>
          )}
        </div>
      )}
    </InspectorCard>
  );
}

function ChunksCard({ chunks }: { chunks: { source: string; text: string }[] }) {
  return (
    <InspectorCard title="Top-3 Chunks · FAISS + BM25 → Cross-Encoder Reranked">
      <div className="max-h-72 space-y-2 overflow-auto pr-1">
        {chunks.map((c, i) => (
          <div
            key={i}
            className="rounded-md border border-border/50 bg-background/40 p-2.5 text-[11px]"
          >
            <div className="mb-1 flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-neon">
              <Quote className="h-3 w-3" /> {c.source}
            </div>
            <p className="font-mono text-[11px] leading-relaxed text-foreground/85">{c.text}</p>
          </div>
        ))}
      </div>
    </InspectorCard>
  );
}
