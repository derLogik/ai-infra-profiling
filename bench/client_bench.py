#!/usr/bin/env python3
"""OpenAI 兼容后端的流式压测客户端（vLLM / SGLang 共用）。

测量口径：
  TTFT  = 请求发出 -> 第一个非空 content delta
  ITL   = (结束时间 - TTFT) / (completion_tokens - 1)
  E2E   = 请求发出 -> 流结束
  吞吐  = 该并发档全部请求的 completion_tokens 之和 / 墙钟时间

只用 httpx + SSE，不依赖 openai SDK 的版本差异。

数据完整性约定（2026-10 修订）
-----------------------------
1. 每个请求的结果都保留，不再只留中位数。逐轮、逐请求写进 JSON，
   这样 min/max/p95、成功率、失败原因都能事后核验。
2. 单个请求失败（HTTP 4xx/5xx、超时、连接错误、流内 error）
   不再中断整轮压测，只记一条失败记录并继续。
3. 并发档同时报告"请求级中位数"和"请求级最大值"。8 个并发请求里
   中位数和最大值是两个不同的量，不能用一个替代另一个。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

import httpx

# (名字, 目标输入 token 数, 输出 token 数)
WORKLOADS = [
    ("short", 128, 128),
    ("medium", 512, 256),
    ("long", 2048, 128),
]

PASSAGE = (
    "Retrieval augmented generation combines a retriever with a generator. "
    "The retriever selects passages from an external corpus, the generator "
    "conditions on those passages, and the system returns an answer. "
    "Serving such a system requires balancing prefill latency, decode "
    "throughput, key value cache memory, and request concurrency. "
)

NAN = float("nan")


def build_prompt(tokenizer, target_tokens: int) -> str:
    """用 tokenizer 实测长度，拼到接近目标 token 数。"""
    text = ""
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += PASSAGE
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)


def _failed(rec: dict, message: str, kind: str, t0: float, status: int | None = None) -> dict:
    """把一次失败写进请求记录，不抛异常。"""
    rec["error"] = message
    rec["error_type"] = kind
    rec["http_status"] = status
    rec["e2e_s"] = time.perf_counter() - t0
    return rec


async def one_request(client, url, model, prompt, max_tokens) -> dict:
    """发一个流式请求。任何失败都以记录形式返回，不向上抛。"""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        # Qwen3 默认开思考模式，128 个 token 可能全花在 reasoning_content 上，
        # 正文 delta 一个都不出现，TTFT 就测不到了。统一关掉，比的是纯解码。
        "chat_template_kwargs": {"enable_thinking": False},
    }
    rec = {
        "ttft_s": NAN,
        "itl_s": NAN,
        "e2e_s": NAN,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "stream_steps": 0,
        "error": None,
        "error_type": None,
        "http_status": None,
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    steps = 0

    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code >= 400:
                # SGLang 在 KV 不够时直接 400（Input length exceeds ...）。
                # 这属于"实验结论的一部分"，必须记下来而不是让整轮崩掉。
                raw = await resp.aread()
                msg = raw.decode("utf-8", errors="replace")[:300].replace("\n", " ")
                return _failed(rec, f"HTTP {resp.status_code}: {msg}", "http",
                               t0, resp.status_code)

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        steps += 1

    except httpx.TimeoutException as exc:
        return _failed(rec, f"timeout: {exc!r}"[:300], "timeout", t0)
    except httpx.HTTPError as exc:
        return _failed(rec, f"transport: {exc!r}"[:300], "transport", t0)
    except Exception as exc:                      # noqa: BLE001 - 记录后继续
        return _failed(rec, f"{type(exc).__name__}: {exc}"[:300], "unexpected", t0)

    t_end = time.perf_counter()
    completion_tokens = (usage or {}).get("completion_tokens") or steps
    prompt_tokens = (usage or {}).get("prompt_tokens") or 0
    rec.update({
        "e2e_s": t_end - t0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "stream_steps": steps,
    })
    if t_first is None:
        return _failed(rec, "no content delta received", "no_content", t0)

    rec["ttft_s"] = t_first - t0
    if completion_tokens > 1:
        rec["itl_s"] = (t_end - t_first) / (completion_tokens - 1)
    return rec


def _clean(values) -> list[float]:
    """去掉 None / NaN，用于统计。"""
    return [v for v in values if v is not None and isinstance(v, (int, float))
            and not math.isnan(v)]


def _median(values) -> float:
    v = _clean(values)
    return statistics.median(v) if v else NAN


def _min(values) -> float:
    v = _clean(values)
    return min(v) if v else NAN


def _max(values) -> float:
    v = _clean(values)
    return max(v) if v else NAN


def _p95(values) -> float:
    """nearest-rank p95；样本少时等于最大值，这是有意为之——宁保守不乐观。"""
    v = sorted(_clean(values))
    if not v:
        return NAN
    k = max(0, math.ceil(0.95 * len(v)) - 1)
    return v[k]


def summarize_cell(samples: list[dict]) -> dict:
    """把一轮压测的所有重复汇总成一个格子。

    `samples` 是逐轮记录，每轮含 `wall_s` 和 `requests`（逐请求记录）。
    纯函数，不碰网络，便于用 mock 数据单测。

    同时给出两套 TTFT 统计，这是本文件最重要的约定：
      - 请求级（ttft_s_median / _max / _p95）：跨所有重复的**所有请求**，
        回答"排队的请求各自等了多久"，用来判断排队/批处理行为；
      - 轮级（ttft_s 等）：每轮取请求中位数、再跨轮取中位数，
        是给表格用的汇总值，不能拿来论证"最后一个请求等了多久"。
    """
    all_req: list[dict] = []
    per_repeat: list[dict] = []
    for s in samples:
        reqs = s.get("requests") or []
        all_req.extend(reqs)
        ok = [r for r in reqs if not r.get("error")]
        out_tokens = sum(r.get("completion_tokens") or 0 for r in reqs)
        wall = s.get("wall_s") or 0.0
        per_repeat.append({
            "wall_s": wall,
            "throughput_tok_s": (out_tokens / wall) if wall > 0 else 0.0,
            "ttft_s": _median([r.get("ttft_s") for r in ok]),
            "itl_s": _median([r.get("itl_s") for r in ok]),
            "e2e_s": _median([r.get("e2e_s") for r in ok]),
            "requests_ok": len(ok),
            "requests_failed": len(reqs) - len(ok),
        })

    ok_req = [r for r in all_req if not r.get("error")]
    failed = [r for r in all_req if r.get("error")]
    error_types: dict[str, int] = {}
    for r in failed:
        key = r.get("error_type") or "unknown"
        error_types[key] = error_types.get(key, 0) + 1

    total = len(all_req)
    if total == 0:
        status = "error"
    elif not failed:
        status = "ok"
    elif ok_req:
        status = "partial"
    else:
        status = "error"

    cell = {
        "repeats": len(samples),
        "requests_total": total,
        "requests_ok": len(ok_req),
        "requests_failed": len(failed),
        "success_rate": (len(ok_req) / total) if total else 0.0,
        "status": status,
        # 轮级汇总（表格用；向后兼容旧字段名）
        "throughput_tok_s": _median(p["throughput_tok_s"] for p in per_repeat),
        "throughput_tok_s_min": _min(p["throughput_tok_s"] for p in per_repeat),
        "throughput_tok_s_max": _max(p["throughput_tok_s"] for p in per_repeat),
        "ttft_s": _median(p["ttft_s"] for p in per_repeat),
        "ttft_s_min": _min(p["ttft_s"] for p in per_repeat),
        "ttft_s_max": _max(p["ttft_s"] for p in per_repeat),
        "itl_s": _median(p["itl_s"] for p in per_repeat),
        "e2e_s": _median(p["e2e_s"] for p in per_repeat),
        # 请求级统计（判断排队行为用）
        "req_ttft_s_median": _median([r.get("ttft_s") for r in ok_req]),
        "req_ttft_s_max": _max([r.get("ttft_s") for r in ok_req]),
        "req_ttft_s_p95": _p95([r.get("ttft_s") for r in ok_req]),
        "req_itl_s_median": _median([r.get("itl_s") for r in ok_req]),
    }
    if error_types:
        cell["error_types"] = error_types
        cell["error_samples"] = [r["error"] for r in failed[:3]]
    cell["samples"] = per_repeat + [{"requests": all_req}]
    return cell


async def run_cell(base_url, model, prompt, max_tokens, concurrency, repeats,
                   timeout_s, max_error_rate=1.0) -> dict:
    """跑一个 (负载, 并发) 格子。

    `max_error_rate`：失败请求占比超过该值时把整格标成 error，
    但仍写完数据——不静默丢弃，也不让一次 400 毁掉整轮矩阵。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        # 预热，让 CUDA graph / 编译缓存就绪，不计入统计
        await one_request(client, url, model, prompt, min(max_tokens, 16))

        samples = []
        for _ in range(repeats):
            wall0 = time.perf_counter()
            results = await asyncio.gather(
                *[one_request(client, url, model, prompt, max_tokens)
                  for _ in range(concurrency)]
            )
            wall = time.perf_counter() - wall0
            samples.append({"wall_s": wall, "requests": list(results)})

    cell = summarize_cell(samples)
    cell["concurrency"] = concurrency
    if cell["success_rate"] == 0.0:
        cell["status"] = "error"
    elif cell["success_rate"] < (1.0 - max_error_rate):
        cell["status"] = "error"
    return cell


async def main_async(args) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    concurrency_levels = [int(x) for x in args.concurrency.split(",")]
    wanted = {w.strip() for w in args.workloads.split(",") if w.strip()}

    results = []
    for name, in_tokens, out_tokens in WORKLOADS:
        if name not in wanted:
            continue
        prompt = build_prompt(tokenizer, in_tokens)
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        for conc in concurrency_levels:
            print(f"  {name:6s} in={actual:5d} out={out_tokens:4d} conc={conc:2d} ...",
                  end="", flush=True)
            cell = await run_cell(
                args.base_url, args.model, prompt, out_tokens,
                conc, args.repeats, args.timeout, args.max_error_rate,
            )
            cell.update({"workload": name, "input_tokens": actual,
                         "output_tokens": out_tokens})
            results.append(cell)
            sr = cell["success_rate"] * 100
            print(f" {cell['throughput_tok_s']:8.1f} tok/s  "
                  f"TTFT {cell['ttft_s']*1000:7.1f} ms  "
                  f"ITL {cell['itl_s']*1000:6.1f} ms  "
                  f"ok {sr:5.1f}%  [{cell['status']}]", flush=True)

    return {
        "backend": args.backend,
        "model": args.model,
        "base_url": args.base_url,
        "version": args.version,
        "results": results,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--workloads", default="short,medium,long")
    p.add_argument("--max-error-rate", type=float, default=1.0,
                   help="失败请求占比超过该值则整格标记 error（默认 1.0 不触发）")
    p.add_argument("--version", default=None, help="后端版本号，写进结果便于追溯")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"[{args.backend}] {args.base_url} model={args.model}", flush=True)
    payload = asyncio.run(main_async(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
