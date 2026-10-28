#!/usr/bin/env python3
"""Turn results/<backend>.json into tables and charts.

Chart labels are English on purpose: the WSL matplotlib environment has no CJK
font, and tofu boxes in a chart are worse than English axis labels.

口径（2026-10 修订）
------------------
每个数值尽量写成 `中位数 [最小 - 最大]`，因为只有中位数时无法判断
两组数字的差异是否在噪声以内。旧结果文件没有 min/max 字段，
这里做了兼容：缺字段就只显示中位数，不伪造区间。

同时输出一节"数据质量"：每格的请求数、失败数、失败类型，
以及不是 ok 的格子。实验可不可信，先看这一节。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BACKENDS = ["transformers", "vllm", "sglang"]
LABELS = {"transformers": "transformers (HF generate)",
          "vllm": "vLLM", "sglang": "SGLang"}
COLORS = {"transformers": "#c0392b", "vllm": "#2e86c1", "sglang": "#27ae60"}
WORKLOADS = ["short", "medium", "long"]
METRICS = [("throughput_tok_s", "Output throughput (tok/s)", False),
           ("ttft_s", "TTFT (ms)", True),
           ("itl_s", "Inter-token latency (ms)", True)]


def load() -> dict[str, dict]:
    out = {}
    for b in BACKENDS:
        p = RESULTS / f"{b}.json"
        if p.exists():
            out[b] = json.loads(p.read_text(encoding="utf-8"))
    return out


def label(data: dict, b: str) -> str:
    """版本号从结果文件里取，不再手写在代码里。"""
    v = (data.get(b) or {}).get("version")
    return f"{LABELS[b]} {v}" if v else LABELS[b]


def fmt(v, scale: float = 1.0) -> str:
    """NaN -> em dash."""
    if v is None or v != v:
        return "—"
    return f"{v * scale:.1f}"


def fmt_range(cell: dict, key: str, scale: float = 1.0) -> str:
    """`中位数 [最小 - 最大]`；没有区间数据时退回单值。"""
    med = cell.get(key)
    lo, hi = cell.get(f"{key}_min"), cell.get(f"{key}_max")
    base = fmt(med, scale)
    if base == "—" or lo is None or hi is None or lo != lo or hi != hi:
        return base
    return f"{base} [{fmt(lo, scale)} - {fmt(hi, scale)}]"


def cells(data: dict) -> dict[tuple[str, str, int], dict]:
    """(backend, workload, concurrency) -> result row"""
    return {(b, r["workload"], r["concurrency"]): r
            for b, payload in data.items() for r in payload["results"]}


def chart(data: dict) -> Path:
    cb = cells(data)
    concs = sorted({k[2] for k in cb})
    fig, axes = plt.subplots(len(WORKLOADS), len(METRICS),
                             figsize=(13, 10), squeeze=False)
    for row, wl in enumerate(WORKLOADS):
        for col, (key, mlabel, convert_ms) in enumerate(METRICS):
            ax = axes[row][col]
            for b in data:
                xs, ys = [], []
                for c in concs:
                    r = cb.get((b, wl, c))
                    if not r or r.get("status") == "oom" or key not in r:
                        continue
                    xs.append(c)
                    ys.append(r[key] * 1000 if convert_ms else r[key])
                if xs:
                    ax.plot(xs, ys, marker="o", label=label(data, b), color=COLORS[b])
            ax.set_title(f"{wl}  ·  {mlabel}", fontsize=10)
            ax.set_xlabel("concurrency / batch", fontsize=9)
            ax.set_xticks(concs)
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=8)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Qwen3-8B-AWQ inference backends on RTX 3070 Laptop (8 GB)", fontsize=13)
    fig.text(0.5, 0.005,
             "x = concurrent requests (vLLM, SGLang) / static batch size "
             "(transformers). transformers batch>1 is wall-clock throughput "
             "only -- no per-token timing is defined for a static batch.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = RESULTS / "backend_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def table(data: dict) -> str:
    cb = cells(data)
    concs = sorted({k[2] for k in cb})
    lines = []
    for wl in WORKLOADS:
        lines.append(f"\n### workload `{wl}`\n")
        head = ("| backend | conc | TTFT (ms) | ITL (ms) | E2E (s) | "
                "throughput (tok/s) | ok% | peak VRAM (GiB) |")
        lines.append(head)
        lines.append("|---|---|---|---|---|---|---|---|")
        for b in data:
            for c in concs:
                r = cb.get((b, wl, c))
                if not r:
                    continue
                if r.get("status") == "oom":
                    lines.append(f"| {label(data, b)} | {c} | OOM | — | — | — | — | — |")
                    continue
                ok = r.get("success_rate")
                oks = f"{ok * 100:.0f}%" if isinstance(ok, (int, float)) else "—"
                lines.append(
                    f"| {label(data, b)} | {c} | {fmt_range(r, 'ttft_s', 1000)} | "
                    f"{fmt(r.get('itl_s'), 1000)} | {fmt(r.get('e2e_s'))} | "
                    f"{fmt_range(r, 'throughput_tok_s')} | {oks} | "
                    f"{fmt(r.get('peak_vram_gib'), 1) if r.get('peak_vram_gib') is not None else '—'} |"
                )
    return "\n".join(lines)


def startup_table(data: dict) -> str:
    lines = ["| backend | version | startup (s) | VRAM after load (GiB) |",
             "|---|---|---|---|"]
    for b, payload in data.items():
        lines.append(f"| {LABELS[b]} | {payload.get('version', '—')} | "
                     f"{payload.get('startup_s', '—')} | {payload.get('vram_gib', '—')} |")
    return "\n".join(lines)


def quality(data: dict) -> str:
    """数据质量：失败请求数、失败类型、非 ok 的格子。"""
    lines = ["| backend | 请求总数 | 失败 | 成功率 | 失败类型 | 非 ok 格子 |",
             "|---|---|---|---|---|---|"]
    for b, payload in data.items():
        rows = payload.get("results", [])
        total = sum(r.get("requests_total", 0) for r in rows)
        failed = sum(r.get("requests_failed", 0) for r in rows)
        rate = (1 - failed / total) * 100 if total else None
        types: dict[str, int] = {}
        for r in rows:
            for k, v in (r.get("error_types") or {}).items():
                types[k] = types.get(k, 0) + v
        bad = [f"{r['workload']}/c{r['concurrency']}({r.get('status')})"
               for r in rows if r.get("status") not in (None, "ok")]
        lines.append(
            f"| {LABELS[b]} | {total or '—'} | {failed} | "
            f"{f'{rate:.1f}%' if rate is not None else '—'} | "
            f"{', '.join(f'{k}×{v}' for k, v in types.items()) or '—'} | "
            f"{', '.join(bad) or '—'} |")
    if not any(r.get("requests_total") for p in data.values() for r in p.get("results", [])):
        lines.append("\n> 注：本结果由旧版客户端采集，未记录逐请求明细，"
                     "成功率一栏无法计算。")
    return "\n".join(lines)


def main() -> int:
    data = load()
    if not data:
        raise SystemExit("results/ 下没有结果 JSON")
    png = chart(data)
    md = ["# Results\n",
          "> 数值格式为 `中位数 [最小 - 最大]`，区间来自每次重复的独立测量。\n"
          "> 逐轮、逐请求的原始记录见对应 `*.json` 的 `samples` 字段。\n",
          "## Startup\n", startup_table(data),
          "\n## Benchmark\n", table(data),
          "\n## 数据质量\n", quality(data)]
    (RESULTS / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"-> {png}")
    print(f"-> {RESULTS / 'tables.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
