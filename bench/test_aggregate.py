#!/usr/bin/env python3
"""压测客户端的口径测试（不需要 GPU、不需要起服务）。

分两层：
  1. summarize_cell 是纯函数，直接用构造数据核对统计口径；
  2. one_request 用 httpx.MockTransport 模拟 SSE / 400 / 超时，
     核对失败请求会被记录而不是抛异常。

跑法：``python -m pytest bench/test_aggregate.py -q``
"""
from __future__ import annotations

import asyncio
import json
import math

import httpx
import pytest

from client_bench import one_request, summarize_cell

URL = "http://127.0.0.1:8000/v1/chat/completions"


# --------------------------------------------------------------------------
# 1) 纯函数：统计口径
# --------------------------------------------------------------------------

def _req(ttft, itl=0.015, e2e=1.0, out=128, err=None, err_type=None):
    return {
        "ttft_s": ttft, "itl_s": itl, "e2e_s": e2e,
        "completion_tokens": out if err is None else 0,
        "prompt_tokens": 128, "stream_steps": out,
        "error": err, "error_type": err_type, "http_status": None,
    }


def test_all_success_cell():
    samples = [
        {"wall_s": 2.0, "requests": [_req(0.1), _req(0.12)]},
        {"wall_s": 2.2, "requests": [_req(0.11), _req(0.13)]},
    ]
    cell = summarize_cell(samples)
    assert cell["status"] == "ok"
    assert cell["requests_total"] == 4
    assert cell["requests_failed"] == 0
    assert cell["success_rate"] == 1.0
    # 每轮 2 个请求 × 128 token，吞吐 = 256 / wall
    assert cell["throughput_tok_s"] == pytest.approx(128 * 2 / 2.1, rel=0.2)


def test_partial_failure_is_recorded_not_raised():
    """一个 400 + 一个成功 → 记 partial，成功的那条仍参与统计。"""
    samples = [{
        "wall_s": 1.0,
        "requests": [_req(0.1), _req(math.nan, err="HTTP 400: too long", err_type="http")],
    }]
    cell = summarize_cell(samples)
    assert cell["status"] == "partial"
    assert cell["requests_failed"] == 1
    assert cell["success_rate"] == 0.5
    assert cell["error_types"] == {"http": 1}
    assert "HTTP 400" in cell["error_samples"][0]
    # NaN 不能污染中位数
    assert cell["ttft_s"] == pytest.approx(0.1)


def test_all_failed_cell_is_error():
    samples = [{
        "wall_s": 1.0,
        "requests": [_req(math.nan, err="timeout: x", err_type="timeout"),
                     _req(math.nan, err="timeout: x", err_type="timeout")],
    }]
    cell = summarize_cell(samples)
    assert cell["status"] == "error"
    assert cell["success_rate"] == 0.0
    assert math.isnan(cell["ttft_s"])


def test_median_and_max_are_reported_separately():
    """8 并发排队时，请求级中位数 != 最大值。

    这是 §6.3 那条结论的口径基础：串行排队下中位数约为最大值的一半，
    如果两者接近，说明请求是"一起等到"而不是"一个一个来"。
    """
    ttfts = [1.0, 1.0, 3.9, 3.9, 6.8, 6.8, 9.7, 9.7]      # 串行排队的形状
    samples = [{"wall_s": 10.0, "requests": [_req(t) for t in ttfts]}]
    cell = summarize_cell(samples)
    assert cell["ttft_s"] == pytest.approx(5.35)          # 轮级中位数
    assert cell["req_ttft_s_median"] == pytest.approx(5.35)
    assert cell["req_ttft_s_max"] == pytest.approx(9.7)
    assert cell["req_ttft_s_p95"] == pytest.approx(9.7)   # 小样本时 p95 == max


def test_per_request_records_are_preserved():
    """逐请求原始记录必须留在结果里，否则事后无法核验。"""
    samples = [{"wall_s": 1.0, "requests": [_req(0.1), _req(0.2)]}]
    cell = summarize_cell(samples)
    assert len(cell["samples"]) == 2
    assert len(cell["samples"][-1]["requests"]) == 2
    assert cell["samples"][-1]["requests"][1]["ttft_s"] == 0.2


def test_empty_cell():
    cell = summarize_cell([])
    assert cell["status"] == "error"
    assert cell["requests_total"] == 0


# --------------------------------------------------------------------------
# 2) HTTP 层：失败路径
# --------------------------------------------------------------------------

def _sse_body(tokens=3, include_usage=True):
    lines = []
    for _ in range(tokens):
        lines.append("data: " + json.dumps(
            {"choices": [{"delta": {"content": "x"}}]}))
    if include_usage:
        lines.append("data: " + json.dumps(
            {"choices": [], "usage": {"completion_tokens": tokens,
                                      "prompt_tokens": 128}}))
    lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def _run(coro):
    return asyncio.run(coro)


def test_sse_success_measures_ttft_and_itl():
    def handler(request):
        return httpx.Response(200, content=_sse_body(tokens=4),
                              headers={"content-type": "text/event-stream"})

    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await one_request(client, URL, "m", "p", 4)

    rec = _run(go())
    assert rec["error"] is None
    assert rec["completion_tokens"] == 4
    assert rec["prompt_tokens"] == 128
    assert rec["ttft_s"] >= 0
    assert rec["itl_s"] >= 0


def test_http_400_is_recorded_as_failure():
    """SGLang KV 不够时的 400，必须变成一条记录而不是异常。"""
    def handler(request):
        return httpx.Response(400, json={"error": "Input length 2060 exceeds max 1035"})

    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await one_request(client, URL, "m", "p", 8)

    rec = _run(go())
    assert rec["error_type"] == "http"
    assert rec["http_status"] == 400
    assert "1035" in rec["error"]
    assert math.isnan(rec["ttft_s"])


def test_timeout_is_recorded_as_failure():
    def handler(request):
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await one_request(client, URL, "m", "p", 8)

    rec = _run(go())
    assert rec["error_type"] == "timeout"
    assert rec["completion_tokens"] == 0


def test_empty_stream_is_recorded():
    def handler(request):
        return httpx.Response(200, content=b"data: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await one_request(client, URL, "m", "p", 8)

    rec = _run(go())
    assert rec["error_type"] == "no_content"
