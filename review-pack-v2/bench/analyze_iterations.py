#!/usr/bin/env python3
"""Parse vLLM iteration-detail logs and judge the real concurrency structure.

背景：报告 §6.3 里"8 个长 prompt 请求到底是逐个串行、还是 2 路 × 4 波"
原先是从吞吐/TTFT/E2E 三个数**反推**的，属于"待验证"。vLLM 打开
`--enable-logging-iteration-details` 后会逐 iteration 打印

    Iteration(17): 2 context requests, 4096 context tokens,
                   0 generation requests, 0 generation tokens, iteration elapsed ...

这一步就是"每步有几个请求在跑"的直接证据。本脚本把它解析出来，判三种模型：

    serial              一次只跑 1 个（真正的排队）
    kv-limited-batching 并发被 KV 容量卡在 k 个，需要 ceil(n/k) 波   ← §6.3 的假设
    fully-batched       全部请求同时进 batch

用法：
    python bench/analyze_iterations.py logs/iter_long_c8.log --expect-requests 8
    python bench/analyze_iterations.py logs/*.log --expect-requests 8 --json

日志来源（在 WSL 里跑，把日志重定向出来）：
    vllm serve <model> ... --enable-logging-iteration-details 2>&1 | tee iter.log
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

# Iteration(17): 2 context requests, 4096 context tokens, 0 generation requests,
# 0 generation tokens, iteration elapsed time: 2.10 ms (dummy)
LINE_RE = re.compile(
    r"Iteration\((?P<idx>\d+)\):\s*"
    r"(?P<ctx_req>\d+)\s+context requests,\s*"
    r"(?P<ctx_tok>\d+)\s+context tokens,\s*"
    r"(?P<gen_req>\d+)\s+generation requests,\s*"
    r"(?P<gen_tok>\d+)\s+generation tokens"
)


def parse(path: Path) -> list[dict]:
    """抽出所有 iteration 行；其它日志行一律忽略。"""
    steps = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LINE_RE.search(line)
            if not m:
                continue
            if "(dummy)" in line:
                continue                      # DP 空转，不代表真实负载
            steps.append({
                "idx": int(m.group("idx")),
                "ctx_req": int(m.group("ctx_req")),
                "ctx_tok": int(m.group("ctx_tok")),
                "gen_req": int(m.group("gen_req")),
                "gen_tok": int(m.group("gen_tok")),
            })
    steps.sort(key=lambda s: s["idx"])
    return steps


def summarize(steps: list[dict], expect_requests: int | None = None) -> dict:
    """纯函数：给出并发结构判读，便于单测。"""
    if not steps:
        return {"ok": False, "reason": "没有解析到 iteration 行", "steps": 0}

    concurrency = [s["ctx_req"] + s["gen_req"] for s in steps]
    max_conc = max(concurrency)
    max_ctx = max(s["ctx_req"] for s in steps)
    max_gen = max(s["gen_req"] for s in steps)

    # 连续出现 ctx_req>0 的段算一次"预填事件"，用来数波次
    bursts, in_burst = 0, False
    for s in steps:
        if s["ctx_req"] > 0:
            if not in_burst:
                bursts += 1
                in_burst = True
        else:
            in_burst = False

    n = expect_requests
    out = {
        "ok": True,
        "steps": len(steps),
        "max_concurrency": max_conc,
        "max_ctx_per_step": max_ctx,
        "max_gen_per_step": max_gen,
        "prefill_bursts": bursts,
        "total_ctx_tokens": sum(s["ctx_tok"] for s in steps),
        "total_gen_tokens": sum(s["gen_tok"] for s in steps),
    }

    if n:
        out["expect_requests"] = n
        out["expected_waves"] = math.ceil(n / max_conc) if max_conc else None
        if max_conc <= 1:
            model = "serial"
        elif max_ctx >= n:
            model = "fully-batched"
        elif max_conc < n:
            model = "kv-limited-batching"
        else:
            model = "fully-batched"
        out["model"] = model
        if model == "kv-limited-batching":
            k = max_conc
            out["inferred_k"] = k
            out["inferred_waves"] = math.ceil(n / k)
            out["matches_2x4"] = (k == 2 and n == 8)
    return out


def timeline(steps: list[dict]) -> str:
    """把 (ctx_req, gen_req) 序列做游程压缩，方便人眼扫一遍。"""
    if not steps:
        return "(空)"
    runs, cur, count = [], (steps[0]["ctx_req"], steps[0]["gen_req"]), 1
    for s in steps[1:]:
        pair = (s["ctx_req"], s["gen_req"])
        if pair == cur:
            count += 1
        else:
            runs.append((cur, count))
            cur, count = pair, 1
    runs.append((cur, count))
    return "  ".join(f"ctx{c}/gen{g}×{n}" for (c, g), n in runs)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+", help="vLLM 日志文件")
    p.add_argument("--expect-requests", type=int, default=None,
                   help="该组负载的并发请求数（例如 8），用于判波次")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    worst = 0
    for raw in args.logs:
        path = Path(raw)
        steps = parse(path)
        res = summarize(steps, args.expect_requests)
        res["file"] = str(path)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        else:
            print(f"\n=== {path.name} ===")
            if not res["ok"]:
                print(f"  {res['reason']}")
                worst = max(worst, 1)
                continue
            print(f"  iteration 步数        : {res['steps']}")
            print(f"  同时在跑的请求数峰值  : {res['max_concurrency']}")
            print(f"  单步 prefill 请求峰值 : {res['max_ctx_per_step']}")
            print(f"  单步 decode 请求峰值  : {res['max_gen_per_step']}")
            print(f"  预填事件次数          : {res['prefill_bursts']}")
            print(f"  ctx / gen token 合计  : {res['total_ctx_tokens']} / {res['total_gen_tokens']}")
            if "model" in res:
                print(f"  判读                  : {res['model']}"
                      + (f"（k={res.get('inferred_k')}，"
                         f"{res.get('inferred_waves')} 波）"
                         if res["model"] == "kv-limited-batching" else ""))
                if res.get("matches_2x4"):
                    print("  → 与 §6.3 的「2 路 × 4 波」假设一致")
            print(f"  时间线                : {timeline(steps)}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
