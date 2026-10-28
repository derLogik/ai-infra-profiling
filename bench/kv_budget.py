#!/usr/bin/env python3
"""KV 容量模型：把显存账本从"手算一次"变成"可执行、可检验、可预测"。

为什么需要它：
手算能得出结论，但不能**预测**。面试里被问"把 gpu-memory-utilization
改成 0.88 会怎样"，手算过的那组数答不了，只有模型能答；而模型一旦
写成代码，它的预测就可以被一次实验证伪。

两条使用方式：

1. 反推（校验模型）：给定日志里看到的 KV 池大小，算出块数和 token 数，
   和日志打印的 `GPU KV cache size` 对得上，说明模型没写错。

    python bench/kv_budget.py --kv-mib 801 --block-size 16

2. 预测（接受检验）：给定 `--gpu-mem-util`，直接算出 KV 池、块数、
   并发上限和波次，然后拿实际跑一次去对。偏差超过 ±5% 就说明
   模型里的 `non_kv` 拟合值漂了。

    python bench/kv_budget.py --gpu-mem-util 0.88

一个必须说清的边界：`non_kv` 是**拟合值**，不是从源码逐项算出来的。
它把所有非 KV 开销（激活峰值、CUDA context、通信缓冲、CUDA graph 估算）
打包成一个数。换模型、换版本、换卡都会变——所以模型给的是
"预测 + 误差带"，不是"精确值"。
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

MIB = 2 ** 20

# 这台机器的实测/拟合参数（Qwen3-8B-AWQ, RTX 3070 Laptop 8 GB, vLLM 0.29.0）
DEVICE_TOTAL_MIB = 8192
WEIGHTS_MIB = 5816          # 5.68 GiB，AWQ 4-bit group_size 128
NON_KV_MIB = 6162.2        # 拟合值：所有非 KV 开销打包
# 拟合依据：0.85 × 8192 = 6963.2 MiB，日志显示 KV 池 801.0 MiB
#   → non_kv = 6963.2 − 801.0 = 6162.2 MiB
# 保留一位小数是有意的：取 6162 会让 0.80 档算出 174 块，
# 而笔记表里是 173 块，差 1 块——这种"差一块"的漂移会污染预测对比。


@dataclass(frozen=True)
class ModelShape:
    """模型结构里决定 KV 大小的那几项。"""
    num_layers: int = 36
    num_kv_heads: int = 8       # GQA：注意不是 32 个注意力头
    head_dim: int = 128
    dtype_bytes: int = 2        # fp16

    def kv_bytes_per_token(self) -> int:
        """单 token 全部层的 KV 字节数：2(K,V) × 层 × 头 × 维 × 字节。"""
        return (2 * self.num_layers * self.num_kv_heads
                * self.head_dim * self.dtype_bytes)


def kv_bytes_per_token(num_layers=36, num_kv_heads=8, head_dim=128,
                       dtype_bytes=2) -> int:
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def available_kv_mib(total_mib=DEVICE_TOTAL_MIB, gpu_mem_util=0.85,
                     non_kv_mib=NON_KV_MIB) -> float:
    requested = total_mib * gpu_mem_util
    return requested - non_kv_mib


def blocks_from_mib(kv_mib: float, block_size: int, shape: ModelShape) -> int:
    """块数 = 可用字节 ÷ 每块的 KV 字节（向下取整，vLLM 也是这么分的）。

    KV 池为负说明显存根本不够（non_kv 已经超过 requested），这时
    块数必须夹到 0 —— 否则会向下游传一个负数块数。
    """
    block_bytes = shape.kv_bytes_per_token() * block_size
    return max(int(kv_mib * MIB // block_bytes), 0)


def usable_blocks(total_blocks: int) -> int:
    """vLLM 会占掉 1 个 null block，真正可用的是 total − 1。"""
    return max(total_blocks - 1, 0)


def blocks_for_request(prompt_tokens: int, completion_tokens: int,
                       block_size: int) -> int:
    """一个请求要占多少块。按完整长度算（vLLM 默认 full-ISL 准入）。"""
    total = prompt_tokens + completion_tokens
    return math.ceil(total / block_size)


def concurrency_estimate(usable: int, per_req_blocks: int) -> dict:
    """按块粒度能同时装下几个完整请求，以及剩余多少块。"""
    if per_req_blocks <= 0:
        return {"concurrency": 0, "used": 0, "free": usable, "utilisation": 0.0}
    k = usable // per_req_blocks
    used = k * per_req_blocks
    free = usable - used
    return {
        "concurrency": k,
        "used": used,
        "free": free,
        "free_tokens": free * 16,
        "utilisation": used / usable if usable else 0.0,
        "short_by": (per_req_blocks - free) if free < per_req_blocks else 0,
    }


def report(total_mib=DEVICE_TOTAL_MIB, gpu_mem_util=0.85,
           non_kv_mib=NON_KV_MIB, block_size=16,
           shape: ModelShape | None = None,
           prompt_tokens=2048, completion_tokens=128,
           max_model_len=4096, requests=8) -> dict:
    shape = shape or ModelShape()
    kv_mib = available_kv_mib(total_mib, gpu_mem_util, non_kv_mib)
    total_blocks = blocks_from_mib(kv_mib, block_size, shape)
    usable = usable_blocks(total_blocks)
    kv_tokens = total_blocks * block_size
    per_req = blocks_for_request(prompt_tokens, completion_tokens, block_size)
    conc = concurrency_estimate(usable, per_req)
    return {
        "gpu_mem_util": gpu_mem_util,
        "kv_mib": round(kv_mib, 1),
        "total_blocks": total_blocks,
        "usable_blocks": usable,
        "kv_tokens": kv_tokens,
        "max_concurrency_log": round(kv_tokens / max_model_len, 2),
        "blocks_per_request": per_req,
        "concurrency": conc["concurrency"],
        "utilisation": round(conc["utilisation"], 3),
        "free_blocks": conc["free"],
        "waves": math.ceil(requests / conc["concurrency"]) if conc["concurrency"] else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="KV 容量模型：反推校验或预测新配置")
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--kv-mib", type=float, default=None,
                   help="直接给 KV 池大小（用于反推校验，忽略 --gpu-mem-util）")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--non-kv-mib", type=float, default=NON_KV_MIB)
    p.add_argument("--prompt-tokens", type=int, default=2048)
    p.add_argument("--completion-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--requests", type=int, default=8)
    args = p.parse_args()

    shape = ModelShape()
    if args.kv_mib is not None:
        kv_mib = args.kv_mib
        util = "(反推)"
    else:
        kv_mib = available_kv_mib(8192, args.gpu_mem_util, args.non_kv_mib)
        util = f"{args.gpu_mem_util}"

    total_blocks = blocks_from_mib(kv_mib, args.block_size, shape)
    usable = usable_blocks(total_blocks)
    kv_tokens = total_blocks * args.block_size
    per_req = blocks_for_request(args.prompt_tokens, args.completion_tokens,
                                 args.block_size)
    conc = concurrency_estimate(usable, per_req)

    print(f"每 token KV        : {shape.kv_bytes_per_token() // 1024} KiB "
          f"({shape.num_layers} 层 × {shape.num_kv_heads} 头 × {shape.head_dim} 维 × 2)")
    print(f"gpu_mem_util       : {util}")
    print(f"KV 池              : {kv_mib:.1f} MiB")
    print(f"块数               : {total_blocks}  (可用 {usable}，扣 1 个 null block)")
    print(f"KV token           : {kv_tokens:,}")
    print(f"max_concurrency    : {kv_tokens / args.max_model_len:.2f}x "
          f"(÷ max_model_len {args.max_model_len})")
    print(f"单请求 {args.prompt_tokens}+{args.completion_tokens} = "
          f"{args.prompt_tokens + args.completion_tokens} token → {per_req} 块")
    print(f"并发               : {conc['concurrency']}  "
          f"(利用率 {conc['utilisation']:.1%}，剩 {conc['free']} 块"
          f" = {conc['free'] * args.block_size:,} token)")
    if conc["short_by"]:
        print(f"                      差 {conc['short_by']} 块就装得下下一个")
    if conc["concurrency"]:
        print(f"{args.requests} 个请求 → {math.ceil(args.requests / conc['concurrency'])} 波")
    return 0


if __name__ == "__main__":
    sys.exit(main())
