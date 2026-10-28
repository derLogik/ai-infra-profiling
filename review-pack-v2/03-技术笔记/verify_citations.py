#!/usr/bin/env python3
"""Check that every file:line citation in the notes still points at the right code.

Why this exists: the notes cite `work/vllm_src`, which is v0.25.0-21
(2026-07-10), while the benchmark actually runs vLLM 0.29.0. Line numbers
drift between versions. A citation that silently rots is worse than no
citation, because it looks authoritative in an interview.

So: every claim that says "see file.py:1234" gets an entry here, and this
script re-reads those lines and matches a fragment. Run it after touching
the notes, after updating vllm_src, or before sending anything to a reviewer.

Usage:
    python notes/verify_citations.py            # verify
    python notes/verify_citations.py -v         # also print the matched line

Exit code is 1 if any citation is stale, 0 if all match.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # ai-infra-profiling/
REPO = ROOT.parent                                     # work/
VLLM = REPO / "vllm_src"
_ORIG_VLLM = VLLM          # 记住默认根，便于 --source-root 重算路径

# (source file, line number, fragment that must appear on that line, why)
CITATIONS: list[tuple[Path, int, str, str]] = [
    # --- kv-cache-allocation.md ---
    (VLLM / "vllm/v1/worker/gpu_worker.py", 498,
     "non_kv_cache_memory", "non-KV 占用 = 三项相加"),
    (VLLM / "vllm/v1/worker/gpu_worker.py", 527,
     "available_kv_cache_memory_bytes", "可用 KV = requested − non_kv − cudagraph"),
    (VLLM / "vllm/v1/worker/gpu_worker.py", 547,
     "Available KV cache memory", "日志那一行的出处"),
    (VLLM / "vllm/v1/core/kv_cache_utils.py", 1001,
     "available_memory // page_size // num_layers", "num_blocks 公式"),
    (VLLM / "vllm/v1/core/kv_cache_utils.py", 2164,
     "get_kv_cache_capacity", "num_tokens / max_concurrency 出处"),
    (VLLM / "vllm/envs.py", 286,
     "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = True", "graph 估算默认开"),
    # --- kv-cache-block-pool.md ---
    (VLLM / "vllm/v1/core/kv_cache_utils.py", 118,
     "class KVCacheBlock", "KVCacheBlock 定义"),
    (VLLM / "vllm/v1/core/kv_cache_utils.py", 153,
     "assert self.block_hash is None", "主 hash 只能设一次"),
    (VLLM / "vllm/v1/core/kv_cache_utils.py", 179,
     "class FreeKVCacheBlockQueue", "自定义双向链表"),
    (VLLM / "vllm/v1/core/block_pool.py", 190,
     "null_block", "null block 从空闲队列摘出"),
    (VLLM / "vllm/v1/core/block_pool.py", 603,
     "_insert_block_hash", "hash 与块多对多"),
    (VLLM / "vllm/v1/core/block_pool.py", 639,
     "popleft_n", "从未命中时从队头取块"),
    (VLLM / "vllm/v1/core/block_pool.py", 657,
     "_maybe_evict_cached_block", "分配前清旧 hash"),
    (VLLM / "vllm/v1/core/block_pool.py", 691,
     "ref_cnt == 0 and not block.is_null", "命中时从空闲队列摘除"),
    (VLLM / "vllm/v1/core/block_pool.py", 717,
     "prepend_n(blocks_without_hash)", "无 hash 块头插（先淘汰）"),
    (VLLM / "vllm/v1/core/block_pool.py", 718,
     "append_n(blocks_with_hash)", "有 hash 块尾插（后淘汰）"),
    (VLLM / "vllm/v1/core/block_pool.py", 749,
     "num_used_blocks != 1", "null block 永远占用"),
    # --- kv-cache-scheduling.md ---
    (VLLM / "vllm/v1/core/kv_cache_manager.py", 269,
     "def allocate_slots", "准入判定入口"),
    (VLLM / "vllm/v1/core/kv_cache_manager.py", 388,
     "watermark_blocks = 0", "watermark 只对 waiting/preempted 生效"),
    (VLLM / "vllm/v1/core/kv_cache_manager.py", 397,
     "if full_sequence_must_fit", "闸 1：整个序列必须装得下"),
    (VLLM / "vllm/v1/core/kv_cache_manager.py", 446,
     "available_blocks", "闸 3：扣掉 reserved 后再比"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 534,
     "while True", "放不下就踢人重试的循环"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 572,
     "self.running.pop()", "默认抢占策略 LIFO"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 576,
     "if preempted_req == request", "可能把自己踢掉"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 1159,
     "num_preemptions += 1", "抢占计数：重算代价的来源"),
    # --- kv-cache-scheduling.md §五修正记录 + §六实验 ---
    (VLLM / "vllm/config/scheduler.py", 140,
     "scheduler_reserve_full_isl: bool = True", "闸 1 默认开"),
    (VLLM / "vllm/config/scheduler.py", 146,
     "watermark: float = Field(default=0.0", "watermark 默认关"),
    (VLLM / "vllm/config/scheduler.py", 153,
     "prefill_schedule_interval", "只对 DP 有意义"),
    (VLLM / "vllm/v1/core/kv_cache_manager.py", 168,
     "self.watermark_blocks = int(watermark", "预留块数 = 比例 × 总块数"),
    (VLLM / "vllm/v1/core/kv_cache_manager.py", 279,
     "full_sequence_must_fit: bool = False", "函数参数默认值，会被调度器覆盖"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 897,
     "reserved_blocks = 0", "默认不预留"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 903,
     "_inflight_prefill_reserved_blocks", "仅异步 KV 加载时非 0"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 914,
     "full_sequence_must_fit=self.scheduler_reserve_full_isl", "配置项在此传入"),
    (VLLM / "vllm/engine/arg_utils.py", 614,
     "scheduler_reserve_full_isl", "暴露为 CLI 参数"),
    (VLLM / "vllm/config/vllm.py", 1609,
     "no-disable-hybrid-kv-cache-manager", "证明 --no- 前缀写法存在"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 416,
     "token_budget = self.max_num_scheduled_tokens", "预算来源"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 112,
     "max_num_batched_tokens", "未设 max_num_scheduled_tokens 时回落"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 480,
     "num_new_tokens = min(num_new_tokens, token_budget)", "chunk 被预算截断"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 108,
     "max_num_running_reqs", "另一道独立上限"),
    # --- kv-cache-scheduling.md §七：抢占 = 重算而非换出 ---
    (VLLM / "vllm/v1/core/sched/scheduler.py", 1152,
     "_free_request_blocks(request)", "抢占时直接归还块，不写回 CPU"),
    (VLLM / "vllm/v1/core/sched/scheduler.py", 1156,
     "num_computed_tokens = 0", "已算 token 清零 → 重排时从头 prefill"),
    (VLLM / "vllm/v1/request.py", 341,
     "PREEMPTED = enum.auto()", "V1 只有 PREEMPTED，没有 SWAPPED"),
]


def check(verbose: bool = False) -> int:
    # 审查包里不含 vLLM 源码树，此时"文件不存在"不是引用过期，
    # 而是根本没法校验。区分这两种情况，否则会把 39 条全报成失效。
    if not VLLM.exists():
        print(f"跳过校验：未找到 vLLM 源码树 {VLLM}")
        print("  这 39 处引用指向 vLLM 源码的 file:line，需要源码树才能核对。")
        print("  如果你拿到的是审查包（不含源码），这是预期行为。")
        print("  要复核：把 vLLM 源码放到 ai-infra-profiling/ 的同级目录 vllm_src/，"
              "或用 --source-root 指定。")
        return 0

    stale = 0
    for path, lineno, frag, why in CITATIONS:
        rel = f"{path.parent.name}/{path.name}"
        if not path.exists():
            print(f"MISSING  {rel}:{lineno}  ({why})")
            stale += 1
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lineno > len(lines):
            print(f"OUT-OF-RANGE  {rel}:{lineno}  文件只有 {len(lines)} 行  ({why})")
            stale += 1
            continue
        got = lines[lineno - 1]
        if frag not in got:
            print(f"STALE  {rel}:{lineno}  ({why})")
            print(f"       期望包含: {frag!r}")
            print(f"       实际是  : {got.strip()[:100]!r}")
            stale += 1
        elif verbose:
            print(f"ok  {rel}:{lineno}  {got.strip()[:78]}")

    total = len(CITATIONS)
    if stale:
        print(f"\n{stale}/{total} 处引用失效 —— 先修笔记再往外发。")
        return 1
    print(f"\n{total} 处引用全部命中。")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--source-root", default=None,
                   help="vLLM 源码树路径（默认 ../vllm_src）；审查包里可用它指向本地副本")
    args = p.parse_args()
    if args.source_root:
        VLLM = Path(args.source_root).resolve()
        # 引用列表是按默认根拼出来的，换根后要重算路径
        CITATIONS[:] = [
            (VLLM / path.relative_to(_ORIG_VLLM) if path.is_relative_to(_ORIG_VLLM)
             else path, lineno, frag, why)
            for path, lineno, frag, why in CITATIONS
        ]
    sys.exit(check(args.verbose))
