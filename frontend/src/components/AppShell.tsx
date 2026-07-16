import { Link, useRouterState } from "@tanstack/react-router";
import { MessageSquare, Database, BarChart2, Sliders, Atom } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { checkHealth } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Q&A Playground", icon: MessageSquare },
  { to: "/knowledge", label: "Knowledge Base", icon: Database },
  { to: "/evaluation", label: "Evaluation", icon: BarChart2 },
  { to: "/tuning", label: "Model Tuning", icon: Sliders },
] as const;

export function AppShell({ children }: { children: ReactNode }) {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    const run = () => checkHealth().then((ok) => alive && setOnline(ok));
    run();
    const id = setInterval(run, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="flex min-h-screen w-full">
      <aside className="glass sticky top-0 flex h-screen w-64 shrink-0 flex-col gap-6 p-5">
        <div className="flex items-center gap-2.5">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-neon-soft neon-glow">
            <Atom className="h-5 w-5 text-neon" strokeWidth={2.25} />
          </div>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-tight">Physics RAG</div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              SLM Console
            </div>
          </div>
        </div>

        <div
          className={cn(
            "flex items-center gap-2 rounded-md px-2.5 py-2 text-xs font-medium",
            online === false ? "bg-destructive/10 text-destructive" : "bg-success/10 text-success"
          )}
        >
          <span
            className={cn(
              "h-2 w-2 rounded-full animate-pulse-dot",
              online === false ? "bg-destructive text-destructive" : "bg-success text-success"
            )}
          />
          {online === false ? "API: Offline" : "API: Online"}
          <span className="ml-auto text-[10px] text-muted-foreground">:8000</span>
        </div>

        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon }) => {
            const active = to === "/" ? path === "/" : path.startsWith(to);
            return (
              <Link
                key={to}
                to={to}
                className={cn(
                  "group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all",
                  active
                    ? "bg-neon-soft text-neon neon-glow"
                    : "text-muted-foreground hover:bg-accent/40 hover:text-foreground"
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="font-medium">{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="mt-auto text-[10px] leading-relaxed text-muted-foreground/70">
          Qwen 2.5-0.5B-Instruct
          <br />
          LoRA · FAISS + BM25
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-6">{children}</main>
    </div>
  );
}
