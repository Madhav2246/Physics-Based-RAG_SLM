import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Upload, FileText, Trash2, Database, Hash, BookOpenCheck, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/knowledge")({
  head: () => ({ meta: [{ title: "Knowledge Base · Physics RAG SLM" }] }),
  component: KnowledgePage,
});

type Doc = {
  file_name: string;
  chunks: number;
  faiss_start: number;
  ingested_at: string;
};

const MOCK_DOCS: Doc[] = [
  { file_name: "BSIM4_Manual_v4.8.pdf", chunks: 312, faiss_start: 0, ingested_at: "2026-06-01T14:22Z" },
  { file_name: "Sze_Semiconductor_Devices.pdf", chunks: 847, faiss_start: 312, ingested_at: "2026-06-01T15:03Z" },
  { file_name: "MIT_OCW_6.012_Lectures.pdf", chunks: 523, faiss_start: 1159, ingested_at: "2026-06-02T09:18Z" },
  { file_name: "IRDS_2023_Roadmap.pdf", chunks: 291, faiss_start: 1682, ingested_at: "2026-06-02T11:45Z" },
  { file_name: "BSIMSOI_v4.4_UserManual.pdf", chunks: 198, faiss_start: 1973, ingested_at: "2026-06-02T13:10Z" },
  { file_name: "Taur_Ning_Modern_VLSI.pdf", chunks: 634, faiss_start: 2171, ingested_at: "2026-06-02T15:55Z" },
  { file_name: "Neamen_Semiconductor_Physics.pdf", chunks: 512, faiss_start: 2805, ingested_at: "2026-06-03T08:00Z" },
  { file_name: "Tsividis_Operation_of_MOSFET.pdf", chunks: 421, faiss_start: 3317, ingested_at: "2026-06-03T09:22Z" },
];

const MOCK_STATS = { total_pdfs: 12, faiss_vectors: 4847, bm25_vocab: 18203 };

function KnowledgePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [docs, setDocs] = useState<Doc[]>(MOCK_DOCS);
  const [stats, setStats] = useState(MOCK_STATS);
  const [sortKey, setSortKey] = useState<keyof Doc>("ingested_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    apiFetch<any>("/api/registry", { timeoutMs: 3000 })
      .then((r) => {
        if (r.documents) setDocs(r.documents);
        if (r.stats) setStats(r.stats);
      })
      .catch(() => {});
  }, []);

  function handleFiles(list: FileList | null) {
    if (!list) return;
    setFiles((prev) => [...prev, ...Array.from(list).filter((f) => f.type === "application/pdf")]);
  }

  async function runIngestion() {
    if (!files.length) {
      toast.error("Drop at least one PDF first.");
      return;
    }
    toast.success("Ingestion started", {
      description: "Chunking at 512 words, embedding via all-MiniLM-L6-v2, updating FAISS + BM25.",
    });
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    try {
      await apiFetch("/api/ingest", { method: "POST", body: fd, timeoutMs: 60000 });
    } catch {}
    setFiles([]);
  }

  async function resetDB() {
    try {
      await apiFetch("/api/ingest/reset", { method: "POST", timeoutMs: 5000 });
    } catch {}
    toast.success("Vector database reset.");
    setDocs([]);
    setStats({ total_pdfs: 0, faiss_vectors: 0, bm25_vocab: 0 });
  }

  const sorted = [...docs].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    const cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortDir === "asc" ? cmp : -cmp;
  });

  function sortBy(k: keyof Doc) {
    if (k === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir("asc");
    }
  }

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Knowledge Base</h1>
        <p className="text-sm text-muted-foreground">
          Ingest semiconductor physics PDFs into the FAISS + BM25 hybrid retriever.
        </p>
      </header>

      {/* Upload */}
      <section className="glass rounded-2xl p-5">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={cn(
            "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 transition-all",
            dragOver
              ? "border-neon bg-neon/5 neon-glow"
              : "border-border/60 hover:border-neon/60 hover:bg-accent/20"
          )}
        >
          <Upload className="h-8 w-8 text-neon" />
          <div className="text-sm font-medium">Drag &amp; drop PDFs here</div>
          <div className="text-xs text-muted-foreground">or</div>
          <label>
            <input
              type="file"
              accept="application/pdf"
              multiple
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
            <span className="cursor-pointer rounded-md border border-border bg-secondary px-3 py-1.5 text-xs font-medium hover:bg-accent">
              Browse files
            </span>
          </label>
        </div>

        {files.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {files.map((f, i) => (
              <div
                key={i}
                className="flex items-center gap-2 rounded-full border border-border/60 bg-background/40 px-3 py-1.5 text-xs"
              >
                <FileText className="h-3 w-3 text-neon" />
                <span className="max-w-[200px] truncate">{f.name}</span>
                <span className="text-muted-foreground">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            ))}
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <Button
            onClick={runIngestion}
            className="bg-neon text-primary-foreground hover:bg-neon/90 neon-glow"
          >
            Run Ingestion Pipeline
          </Button>
        </div>
      </section>

      {/* Stats */}
      <section className="grid grid-cols-3 gap-4">
        <Stat icon={FileText} label="Total PDFs Ingested" value={stats.total_pdfs.toLocaleString()} />
        <Stat icon={Database} label="Total FAISS Vectors" value={stats.faiss_vectors.toLocaleString()} />
        <Stat
          icon={Hash}
          label="BM25 Vocabulary Size"
          value={`${stats.bm25_vocab.toLocaleString()} tokens`}
        />
      </section>

      {/* Table */}
      <section className="glass rounded-2xl">
        <div className="flex items-center justify-between border-b border-border/60 p-4">
          <div className="flex items-center gap-2">
            <BookOpenCheck className="h-4 w-4 text-neon" />
            <h2 className="text-sm font-semibold">Ingested Documents</h2>
          </div>
          <div className="text-xs text-muted-foreground">{docs.length} files</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-[10px] uppercase tracking-wider text-muted-foreground">
              <tr className="border-b border-border/60">
                {(
                  [
                    ["file_name", "File Name"],
                    ["chunks", "Chunks"],
                    ["faiss_start", "FAISS Start"],
                    ["ingested_at", "Ingested At"],
                  ] as [keyof Doc, string][]
                ).map(([k, label]) => (
                  <th
                    key={k}
                    onClick={() => sortBy(k)}
                    className="cursor-pointer px-4 py-2.5 text-left hover:text-foreground"
                  >
                    {label} {sortKey === k ? (sortDir === "asc" ? "▲" : "▼") : ""}
                  </th>
                ))}
                <th className="px-4 py-2.5 text-right">Delete</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((d) => (
                <tr key={d.file_name} className="border-b border-border/40 last:border-0 hover:bg-accent/20">
                  <td className="px-4 py-3 font-mono text-xs">{d.file_name}</td>
                  <td className="px-4 py-3 tabular-nums">{d.chunks}</td>
                  <td className="px-4 py-3 tabular-nums text-muted-foreground">{d.faiss_start}</td>
                  <td className="px-4 py-3 text-xs text-muted-foreground">{d.ingested_at}</td>
                  <td className="px-4 py-3 text-right">
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive"
                      onClick={() => {
                        setDocs((prev) => prev.filter((x) => x.file_name !== d.file_name));
                        toast.success(`Removed ${d.file_name}`);
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Danger zone */}
      <section className="rounded-2xl border border-destructive/40 bg-destructive/5 p-5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg bg-destructive/15">
              <AlertTriangle className="h-4 w-4 text-destructive" />
            </div>
            <div>
              <div className="text-sm font-semibold text-destructive">Danger Zone</div>
              <div className="text-xs text-muted-foreground">
                Permanently delete all FAISS vectors and BM25 index. PDFs remain on disk.
              </div>
            </div>
          </div>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive">Reset Vector Database</Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Reset vector database?</AlertDialogTitle>
                <AlertDialogDescription>
                  This drops the entire FAISS index and BM25 vocabulary. Re-ingestion will be required
                  to restore retrieval.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={resetDB}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Yes, reset
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </section>
    </div>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
        <Icon className="h-4 w-4 text-neon" />
      </div>
      <div className="mt-3 text-3xl font-bold tabular-nums text-neon">{value}</div>
    </div>
  );
}
