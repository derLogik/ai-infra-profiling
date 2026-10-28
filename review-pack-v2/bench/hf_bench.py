#!/usr/bin/env python3
"""In-process transformers baseline for the same AWQ checkpoint.

Why this exists: vLLM and SGLang are measured over HTTP with client_bench.py.
This script measures the *library baseline* -- plain HF generate() -- so the
three runtimes can be compared on identical weights and identical prompt /
output token counts. transformers has no continuous batching, so "concurrency"
here means a static batch (one generate() call over N copies of the prompt);
that difference is the finding, not a bug.

Measurement definitions match client_bench.py:
  TTFT = generate() call -> first streamed token
  ITL  = (last token time - first token time) / (steps - 1)
  throughput = batch_size * max_new_tokens / wall clock

TTFT / ITL are only meaningful at batch_size == 1. TextIteratorStreamer does
not support batch > 1 (newer versions raise, older ones emit one increment per
sequence per step, inflating the step count N-fold), so batching runs
non-streamed and only wall-clock throughput is reported (ttft/itl = NaN).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

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


def build_prompt(tokenizer, target_tokens: int) -> str:
    text = ""
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += PASSAGE
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids)


def make_inputs(tokenizer, prompt: str, batch_size: int, chat_template: bool):
    """构造模型输入。

    chat_template=True（默认）与 client_bench 对齐：那边发的是
    `/v1/chat/completions` 的 messages，服务端会套 chat template。
    如果这边直接喂原始文本，两边的"2048 token prompt"在模型那里
    根本不是同一个输入，prompt 长度和生成行为都对不上。
    """
    if not chat_template:
        return tokenizer([prompt] * batch_size, return_tensors="pt",
                         padding=True)

    msgs = [{"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", enable_thinking=False,
        )
    except TypeError:
        # 老版本 tokenizer 不认 enable_thinking，退回不带该参数
        ids = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_tensors="pt",
        )
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids.repeat(batch_size, 1)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def one_batch(model, tokenizer, prompt, max_tokens, batch_size, stream,
              chat_template=True):
    from transformers import TextIteratorStreamer

    inputs = make_inputs(tokenizer, prompt, batch_size, chat_template).to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        min_new_tokens=max_tokens,   # ignore_eos 的等价物
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    t0 = time.perf_counter()
    if not stream:
        model.generate(**gen_kwargs)
        t_end = time.perf_counter()
        return {
            "ttft_s": float("nan"),
            "itl_s": float("nan"),
            "e2e_s": t_end - t0,
            "completion_tokens": batch_size * max_tokens,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "requests": batch_size,
            "mode": "batch",
        }

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                    skip_special_tokens=True,
                                    timeout=600.0)
    thread = threading.Thread(target=model.generate,
                              kwargs={**gen_kwargs, "streamer": streamer},
                              daemon=True)
    thread.start()

    t_first = None
    steps = 0
    for _ in streamer:
        if t_first is None:
            t_first = time.perf_counter()
        steps += 1
    thread.join()
    t_end = time.perf_counter()

    # TextIteratorStreamer 每次 yield 的是一段解码后的文本，不严格等于一个
    # token（可能一次吐几个 token，也可能被 skip_special_tokens 合并）。
    # 而 min_new_tokens=max_new_tokens 已经保证"正好生成 max_tokens 个新 token"，
    # 所以 token 数以 max_tokens 为准；steps 只留作诊断，不参与吞吐/ITL 计算。
    # 旧实现拿 steps 当 token 数，会把 ITL 系统性地算偏。
    completion_tokens = max_tokens
    ttft = (t_first - t0) if t_first else float("nan")
    itl = ((t_end - t_first) / (max_tokens - 1)
           if max_tokens > 1 and t_first else float("nan"))
    return {
        "ttft_s": ttft,
        "itl_s": itl,
        "e2e_s": t_end - t0,
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "stream_steps": steps,
        "requests": batch_size,
        "mode": "stream",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="transformers")
    p.add_argument("--model-path", required=True)
    p.add_argument("--batch-sizes", default="1,4")
    p.add_argument("--workloads", default="short,medium,long",
                   help="只跑指定负载，便于单独复测某一格")
    p.add_argument("--repeats", type=int, default=3)
    # gptqmodel 默认选 AwqMarlinLinear，但它要现场 JIT 编译 CUDA 扩展，
    # 在这台机器的 WSL 里每次都把 VM 打崩（Wsl/Service/E_UNEXPECTED）。
    # gemm_triton 走 Triton，不需要编译，代价是 GEMM 内核慢一点。
    p.add_argument("--awq-backend", default="gemm_triton")
    p.add_argument("--chat-template", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="套 chat template，与 client_bench 的 messages 口径对齐；"
                        "关掉则回到裸文本生成（与旧数据一致但不可与引擎侧直接比较）")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"     # 批量生成必须左填充，右填充会错位
    print(f"[transformers] loading {args.model_path} (awq backend={args.awq_backend})",
          flush=True)
    quant_config = AwqConfig(
        bits=4, group_size=128, zero_point=True, version="gemm",
        backend=args.awq_backend,
    )
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        dtype=torch.float16,          # AWQ CUDA kernel 只支持 fp16
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    load_s = time.perf_counter() - t_load
    used = torch.cuda.memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"  loaded in {load_s:.1f}s, VRAM {used:.2f} / {total:.2f} GiB", flush=True)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    order = [w.strip() for w in args.workloads.split(",")]
    wanted = set(order)
    # 按用户给的顺序跑：笔记本 GPU 有时钟爬坡，先跑的那格会系统性偏慢，
    # 需要能调换顺序做对照实验。
    plan = sorted(WORKLOADS,
                  key=lambda w: order.index(w[0]) if w[0] in wanted else 99)

    # 全局预热，全部丢弃。实测：模型刚加载后先跑的那一格会慢 2~2.5 倍，
    # 且与负载无关（long 先跑时 279ms/词，short 后跑只有 124ms/词）。
    # 原因是笔记本 GPU 还处于低频 + Triton 按真实序列长度现场 autotune。
    # 每格自带的 16-token 预热覆盖不到这两者，所以先按真实长度空跑一遍。
    for _name, _in, _out in plan:
        if _name not in wanted:
            continue
        _prompt = build_prompt(tokenizer, _in)
        for _bs in batch_sizes:
            try:
                one_batch(model, tokenizer, _prompt, _out, _bs, _bs == 1, args.chat_template)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    print("  global warmup done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []

    def dump():
        """每格都落盘：AWQ 内核在这台机器上可能把 VM 带崩。"""
        payload = {"backend": args.backend, "model": args.model_path,
                   "startup_s": round(load_s, 2),
                   "vram_gib": round(used, 2),
                   # 基线配置必须落盘：这个文件以前 server_args 是空的，
                   # 导致"比 transformers 快几倍"无法说清是用什么配置测的
                   "bench_config": {
                       "dtype": "float16",
                       "quantization": "awq", "bits": 4, "group_size": 128,
                       "awq_backend": args.awq_backend,
                       "attn_implementation": "sdpa",
                       "chat_template": bool(args.chat_template),
                       "ignore_eos_equivalent": "min_new_tokens=max_new_tokens",
                       "device_map": "cuda",
                   },
                   "results": results}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"  -> {out}", flush=True)
    for name, in_tokens, out_tokens in plan:
        if name not in wanted:
            continue
        prompt = build_prompt(tokenizer, in_tokens)
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        for bs in batch_sizes:
            stream = bs == 1
            label = "stream" if stream else "batch "
            print(f"  {name:6s} in={actual:5d} out={out_tokens:4d} batch={bs:2d} "
                  f"mode={label} ...", end="", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 预热：内核选择 / 分配器预热不计入统计
            try:
                one_batch(model, tokenizer, prompt, min(out_tokens, 16), bs, stream, args.chat_template)
            except torch.OutOfMemoryError:
                print(" OOM (warmup)", flush=True)
                results.append({
                    "backend": args.backend, "workload": name,
                    "input_tokens": actual, "output_tokens": out_tokens,
                    "concurrency": bs, "status": "oom",
                })
                torch.cuda.empty_cache()
                dump()
                continue

            samples = []
            oom = False
            for _ in range(args.repeats):
                try:
                    r = one_batch(model, tokenizer, prompt, out_tokens, bs, stream, args.chat_template)
                except torch.OutOfMemoryError:
                    oom = True
                    torch.cuda.empty_cache()
                    break
                wall = r["e2e_s"]
                samples.append({
                    "throughput_tok_s": r["completion_tokens"] / wall,
                    "ttft_s": r["ttft_s"],
                    "itl_s": r["itl_s"],
                    "e2e_s": r["e2e_s"],
                })

            peak = torch.cuda.max_memory_allocated() / 2**30
            if oom or not samples:
                print(" OOM", flush=True)
                results.append({
                    "backend": args.backend, "workload": name,
                    "input_tokens": actual, "output_tokens": out_tokens,
                    "concurrency": bs, "status": "oom", "peak_vram_gib": round(peak, 2),
                })
                dump()
                continue

            def med(k):
                return statistics.median(s[k] for s in samples)

            def lo(k):
                return min(s[k] for s in samples)

            def hi(k):
                return max(s[k] for s in samples)

            cell = {
                "backend": args.backend,
                "workload": name,
                "input_tokens": actual,
                "output_tokens": out_tokens,
                "concurrency": bs,
                "repeats": len(samples),
                "mode": "stream" if stream else "batch",
                # 逐轮原始样本保留，事后可核验区间与抖动；
                # transformers 是静态批，不存在"逐请求"这一层。
                "samples": samples,
                "requests_total": len(samples) * bs,
                "requests_ok": len(samples) * bs,
                "requests_failed": 0,
                "success_rate": 1.0,
                "status": "ok",
                "throughput_tok_s": med("throughput_tok_s"),
                "throughput_tok_s_min": lo("throughput_tok_s"),
                "throughput_tok_s_max": hi("throughput_tok_s"),
                "ttft_s": med("ttft_s"),
                "ttft_s_min": lo("ttft_s"),
                "ttft_s_max": hi("ttft_s"),
                "itl_s": med("itl_s"),
                "e2e_s": med("e2e_s"),
                "peak_vram_gib": round(peak, 2),
            }
            results.append(cell)
            print(f" {cell['throughput_tok_s']:8.1f} tok/s  "
                  f"TTFT {cell['ttft_s']*1000:7.1f} ms  "
                  f"ITL {cell['itl_s']*1000:6.1f} ms  "
                  f"peak {peak:.2f} GiB", flush=True)
            dump()

    dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
