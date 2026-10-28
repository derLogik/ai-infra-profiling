#!/usr/bin/env python3
"""analyze_iterations 的判读测试（合成日志，不需要 GPU、不需要真日志）。

三种并发模型各造一份日志，确认判读结果符合预期：
  serial              一次 1 个
  kv-limited-batching 2 路 × 4 波（§6.3 的假设）
  fully-batched       8 个同时
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from analyze_iterations import parse, summarize

# 不用 pytest 的 tmp_path：它建在系统临时目录，本机沙箱下不可写。
# 统一落在仓库内的 .testtmp/，跑完即删。
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def tmp_path():
    d = ROOT / ".testtmp" / "iter"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)

TEMPLATE = ("INFO ... Iteration({i}): {c} context requests, {ct} context tokens, "
            "{g} generation requests, {gt} generation tokens, "
            "iteration elapsed time: 12.34 ms")


def _write(tmp_path: Path, steps: list[tuple[int, int, int, int]]) -> Path:
    lines = []
    for i, (c, ct, g, gt) in enumerate(steps):
        lines.append(TEMPLATE.format(i=i, c=c, ct=ct, g=g, gt=gt))
        # 中间插一些无关日志，确认解析器不会被干扰
        if i == 0:
            lines.append("INFO 10-23 07:34:41 [gpu_worker.py:429] Using V2 Model Runner")
    p = tmp_path / "iter.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_serial(tmp_path):
    steps = []
    for _ in range(8):                 # 8 个请求，一个一个来
        steps.append((1, 2048, 0, 0))
        steps.extend([(0, 0, 1, 1)] * 4)
    res = summarize(parse(_write(tmp_path, steps)), expect_requests=8)
    assert res["max_concurrency"] == 1
    assert res["model"] == "serial"


def test_kv_limited_2x4(tmp_path):
    steps = []
    for _ in range(4):                 # 4 波，每波 2 个
        steps.append((2, 4096, 0, 0))
        steps.extend([(0, 0, 2, 2)] * 3)
    res = summarize(parse(_write(tmp_path, steps)), expect_requests=8)
    assert res["max_concurrency"] == 2
    assert res["max_ctx_per_step"] == 2
    assert res["prefill_bursts"] == 4
    assert res["model"] == "kv-limited-batching"
    assert res["inferred_k"] == 2
    assert res["inferred_waves"] == 4
    assert res["matches_2x4"] is True


def test_fully_batched(tmp_path):
    steps = [(8, 16384, 0, 0)] + [(0, 0, 8, 8)] * 5
    res = summarize(parse(_write(tmp_path, steps)), expect_requests=8)
    assert res["max_concurrency"] == 8
    assert res["model"] == "fully-batched"


def test_dummy_iterations_are_ignored(tmp_path):
    p = tmp_path / "iter.log"
    p.write_text(
        TEMPLATE.format(i=0, c=0, ct=0, g=0, gt=0) + " (dummy)\n"
        + TEMPLATE.format(i=1, c=2, ct=4096, g=0, gt=0) + "\n",
        encoding="utf-8")
    steps = parse(p)
    assert len(steps) == 1
    assert steps[0]["ctx_req"] == 2


def test_empty_log_is_reported_not_crashed(tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("INFO nothing here\n", encoding="utf-8")
    res = summarize(parse(p))
    assert res["ok"] is False
