#!/usr/bin/env python3
"""sanitize_results 的单测（纯字符串处理，不需要 GPU）。

只测一件事：**该脱敏的脱掉，不该改的一个字都不动**。
测量值被误改是这个脚本最可怕的失败模式。
"""
from __future__ import annotations

import json

import pytest

from sanitize_results import sanitize_text


def test_strips_linux_home():
    s = '{"model": "/home/alice/models/Qwen3-8B-AWQ"}'
    out, n = sanitize_text(s)
    assert n == 1
    assert out == '{"model": "<HOME>/models/Qwen3-8B-AWQ"}'
    assert "alice" not in out


def test_strips_macos_home():
    out, n = sanitize_text('"/Users/alice/venvs/vllm/bin/vllm"')
    assert n == 1 and out == '"<HOME>/venvs/vllm/bin/vllm"'


def test_strips_windows_home():
    out, n = sanitize_text(r'"C:\Users\alice\work\bench\run.py"')
    assert n == 1
    assert out == r'"<HOME>\\work\bench\run.py"'.replace("\\\\", "\\")
    assert "alice" not in out


def test_multiple_occurrences_counted():
    s = "/home/alice/a /home/alice/b /home/alice/c"
    out, n = sanitize_text(s)
    assert n == 3
    assert out == "<HOME>/a <HOME>/b <HOME>/c"


def test_measurements_untouched():
    """核心保证：数字、指标名、127.0.0.1 这些都不许动。"""
    payload = {
        "ttft_s": 8.089, "itl_ms": 15.63, "throughput_tok_s": 63.4,
        "base_url": "http://127.0.0.1:8000",
        "peak_vram_gib": 7.43,
    }
    before = json.dumps(payload)
    out, n = sanitize_text(before)
    assert n == 0
    assert json.loads(out) == payload          # 逐字段相等，不靠字符串比


def test_localhost_kept():
    out, n = sanitize_text('"127.0.0.1"')
    assert n == 0 and out == '"127.0.0.1"'


def test_no_false_positive_on_relative_path():
    out, n = sanitize_text('"bench/client_bench.py"')
    assert n == 0 and out == '"bench/client_bench.py"'


@pytest.mark.parametrize("s", [
    "/homepage/index.html",       # 不是 /home/<user>/
    "/Users/",                    # 缺用户名
    "/home/alice",                # 缺结尾斜杠
])
def test_does_not_overstrip(s):
    out, n = sanitize_text(f'"{s}"')
    assert n == 0 and out == f'"{s}"'
