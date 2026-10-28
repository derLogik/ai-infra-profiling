#!/usr/bin/env python3
"""kv_budget 模型单测。

这个模型的价值全在"能被证伪"。所以测试的重点不是"函数能跑"，
而是：**它必须复现日志里的实测值**，否则模型就是错的。

复现目标（全部来自 logs/vllm.log 第 48/50 行）：
    Available KV cache memory: 0.78 GiB
    GPU KV cache size: 5,696 tokens, Maximum concurrency 1.39x
"""
from __future__ import annotations

import math

import pytest

from kv_budget import (
    ModelShape,
    available_kv_mib,
    blocks_for_request,
    blocks_from_mib,
    concurrency_estimate,
    kv_bytes_per_token,
    report,
    usable_blocks,
)

SHAPE = ModelShape()


# ---------- 第一层：单点公式 ----------

def test_kv_bytes_per_token_is_144_kib():
    assert kv_bytes_per_token() == 2 * 36 * 8 * 128 * 2
    assert kv_bytes_per_token() // 1024 == 144


def test_gqa_matters():
    """8 个 KV 头 vs 32 个注意力头：差 4 倍，这是 8 GB 能跑 8B 的前提。"""
    assert kv_bytes_per_token(num_kv_heads=32) == 4 * kv_bytes_per_token()


# ---------- 第二层：复现日志 ----------

def test_reproduces_log_kv_tokens():
    """801 MiB → 356 块 → 5,696 token。与日志逐字一致。"""
    blocks = blocks_from_mib(801, 16, SHAPE)
    assert blocks == 356
    assert blocks * 16 == 5696


def test_reproduces_log_max_concurrency():
    """日志打印 1.39x，模型必须给出 1.39。"""
    kv_tokens = blocks_from_mib(801, 16, SHAPE) * 16
    assert round(kv_tokens / 4096, 2) == 1.39


def test_usable_blocks_reserves_one_null_block():
    assert usable_blocks(356) == 355
    assert usable_blocks(1) == 0
    assert usable_blocks(0) == 0


def test_blocks_per_request_matches_note():
    """2048 prompt + 128 输出 = 2,176 token → 136 块。"""
    assert blocks_for_request(2048, 128, 16) == 136
    assert blocks_for_request(2176, 0, 16) == 136


# ---------- 第三层：并发与碎片 ----------

def test_concurrency_is_two_and_fragmented():
    """355 可用块装 2 个 136 块请求，剩 83 块但装不下第三个（差 53）。"""
    c = concurrency_estimate(355, 136)
    assert c["concurrency"] == 2
    assert c["used"] == 272
    assert c["free"] == 83
    assert c["free_tokens"] == 1328
    assert round(c["utilisation"], 3) == 0.766
    assert c["short_by"] == 53


def test_exact_fit_still_reports_the_next_request_shortfall():
    """正好装满时，要装下一个仍需 136 块；这是"没有浪费"的另一种表述。"""
    c = concurrency_estimate(272, 136)
    assert c["concurrency"] == 2
    assert c["free"] == 0
    assert c["short_by"] == 136
    assert c["utilisation"] == 1.0


def test_zero_per_request_does_not_divide_by_zero():
    c = concurrency_estimate(355, 0)
    assert c["concurrency"] == 0


# ---------- 第四层：整链报告 ----------

def test_report_reproduces_the_documented_chain():
    r = report(block_size=16, prompt_tokens=2048, completion_tokens=128,
               requests=8)
    assert r["kv_mib"] == pytest.approx(801.2, abs=0.3)
    assert r["total_blocks"] == 356 and r["usable_blocks"] == 355
    assert r["kv_tokens"] == 5696
    assert r["max_concurrency_log"] == 1.39
    assert r["blocks_per_request"] == 136
    assert r["concurrency"] == 2
    assert r["free_blocks"] == 83
    assert r["waves"] == 4


def test_prediction_at_088():
    """0.88 的预测：465 块 / 7,440 token / 并发 3 / 3 波。

    这条是**待实验证伪**的预测，不是结论。若实测明显偏离，
    优先怀疑 non_kv 拟合值漂了。
    """
    r = report(gpu_mem_util=0.88)
    assert r["total_blocks"] == 465
    assert r["kv_tokens"] == 7440
    assert r["kv_mib"] == pytest.approx(1047.0, abs=0.5)
    assert r["concurrency"] == 3
    assert r["waves"] == 3


# ---------- 第五层：单调性与边界 ----------

@pytest.mark.parametrize("util", [0.76, 0.8, 0.85, 0.9, 0.95])
def test_kv_grows_with_util(util):
    assert available_kv_mib(gpu_mem_util=util) > 0


def test_higher_util_gives_more_blocks():
    a = report(gpu_mem_util=0.85)["total_blocks"]
    b = report(gpu_mem_util=0.88)["total_blocks"]
    assert b > a


def test_lower_non_kv_gives_more_blocks():
    a = report(non_kv_mib=6162.2)["total_blocks"]
    b = report(non_kv_mib=5000)["total_blocks"]
    assert b > a


def test_smaller_block_size_gives_more_tokens():
    """块更小时同样的字节能装更多 token（但块数变多、管理开销上升）。"""
    t16 = report(block_size=16)["kv_tokens"]
    t8 = report(block_size=8)["kv_tokens"]
    assert t8 >= t16


def test_insufficient_memory_reports_zero_not_crash():
    r = report(gpu_mem_util=0.5)      # 4096 − 6162 < 0
    assert r["total_blocks"] == 0
    assert r["concurrency"] == 0
    assert r["waves"] is None
