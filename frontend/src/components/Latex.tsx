import * as katexModule from "react-katex";
const BlockMath: React.ComponentType<{ math: string }> = (katexModule as any).BlockMath || (katexModule as any).default?.BlockMath;
const InlineMath: React.ComponentType<{ math: string }> = (katexModule as any).InlineMath || (katexModule as any).default?.InlineMath;
import type React from "react";

// Renders mixed markdown-ish text with $...$ inline and $$...$$ block math.
export function Latex({ children }: { children: string }) {
  const parts: Array<{ type: "text" | "inline" | "block"; value: string }> = [];
  const re = /\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(children))) {
    if (m.index > last) parts.push({ type: "text", value: children.slice(last, m.index) });
    if (m[1] !== undefined) parts.push({ type: "block", value: m[1] });
    else parts.push({ type: "inline", value: m[2] });
    last = m.index + m[0].length;
  }
  if (last < children.length) parts.push({ type: "text", value: children.slice(last) });

  return (
    <div className="space-y-3 text-[14px] leading-relaxed">
      {parts.map((p, i) => {
        if (p.type === "block")
          return (
            <div key={i} className="overflow-x-auto rounded-md bg-background/40 px-3 py-2">
              <BlockMath math={p.value} />
            </div>
          );
        if (p.type === "inline") return <InlineMath key={i} math={p.value} />;
        return (
          <span key={i} className="whitespace-pre-wrap">
            {p.value}
          </span>
        );
      })}
    </div>
  );
}
