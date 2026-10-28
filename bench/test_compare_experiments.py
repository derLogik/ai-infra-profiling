#!/usr/bin/env python3
"""compare_experiments 的单测（纯计算，不需要 GPU、不需要真实结果）。

覆盖两件最容易出错的事：
1. 判定 —— 实测在容差内算"符合"，超出算"偏离"，缺数据算"无法判定"
2. 缺数据不崩 —— 某组跑挂了也要留在表里，标 `—`，而不是整张表消失
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from compare_experiments import (
    build_arm,
    cell_metrics,
    count_preemptions,
    find_cell,
    render,
    tag_of,
    verdict,
)

ROOT = Path(__file__).resolve().parent


@pytest.fixture()
def tmp_path():
    d = ROOT / ".testtmp" / "cmp"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def payload(rows, peak=7.4):
    return {"peak_vram_gib": peak, "results": rows}


def test_tag_of_with_and_without_suffix():
    assert tag_of(Path("results/vllm__A.json")) == "A"
    assert tag_of(Path("results/vllm.json")) == "vllm"


def test_find_cell_matches_workload_and_concurrency():
    pl = payload([{"workload": "long", "concurrency": 8, "throughput_tok_s": 63.4}])
    assert find_cell(pl, "long", 8)["throughput_tok_s"] == 63.4
    assert find_cell(pl, "long", 4) is None


def test_cell_metrics_extracts_fields():
    pl = payload([{
        "workload": "long", "concurrency": 8, "status": "ok",
        "throughput_tok_s": 63.4, "ttft_s": 8.09,
        "req_ttft_s_p95": 10.5, "success_rate": 1.0,
    }])
    m = cell_metrics(pl, "long", 8)
    assert m["present"] and m["throughput"] == 63.4
    assert m["ttft_median"] == 8.09 and m["ttft_p95"] == 10.5
    assert m["peak_vram"] == 7.4


def test_cell_metrics_missing_returns_not_present():
    m = cell_metrics(payload([]), "long", 8)
    assert m["present"] is False and m.get("throughput") is None


def test_count_preemptions_counts_lines_not_words():
    text = "\n".join([
        "INFO Preemption: seq 3 is preempted",
        "INFO normal step",
        "num_preemptions=2",          # 统计行，不算事件
        "INFO Preemption: seq 5",
    ])
    assert count_preemptions(text) == 2


def test_count_preemptions_zero_when_absent():
    assert count_preemptions("all good\nno problems") == 0


@pytest.mark.parametrize("obs,pred,tol,expect", [
    (63.4, 63.4, 0.15, "符合"),
    (63.4, 60.0, 0.15, "符合"),      # 偏离 5.7%
    (80.0, 63.4, 0.15, "偏离"),      # 偏离 26%
    (None, 63.4, 0.15, "无法判定"),
    (63.4, None, 0.15, "无法判定"),
    (0.0, 0.0, 0.15, "符合"),
    (5.0, 0.0, 0.15, "偏离"),
])
def test_verdict(obs, pred, tol, expect):
    assert verdict(obs, pred, tol) == expect


def test_render_marks_each_arm_and_keeps_missing():
    preds = {"A": {"concurrency": 2, "throughput": 63.4, "note": "基线"},
             "B": {"concurrency": 2, "throughput": 63.4, "note": "预期无变化"}}
    arms = [
        {"tag": "A", "metrics": {"throughput": 63.0, "ttft_median": 8.1},
         "concurrency_obs": 2, "preemptions": 0},
        {"tag": "B", "metrics": {"present": False}, "concurrency_obs": None,
         "preemptions": 0},
    ]
    text = render(arms, preds)
    assert "| A | 2 | 2 |" in text
    assert "符合" in text and "无法判定" in text
    assert "预期无变化" in text
    assert "| B |" in text          # 缺数据的那组仍在表里


def test_build_arm_reads_json_and_tolerates_missing_log(tmp_path):
    jp = tmp_path / "vllm__C.json"
    jp.write_text(json.dumps({
        "peak_vram_gib": 7.5,
        "results": [{"workload": "long", "concurrency": 8,
                     "throughput_tok_s": 71.2, "ttft_s": 6.0}],
    }, ensure_ascii=False), encoding="utf-8")
    arm = build_arm(jp, tmp_path / "nope.log", "long", 8, 8)
    assert arm["tag"] == "C"
    assert arm["metrics"]["throughput"] == 71.2
    assert arm["preemptions"] == 0
    assert arm["concurrency_obs"] is None       # 没日志就没有并发判读


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    """跑一遍真实 CLI：预测 vs 实测的对照表必须出现，且缺数据不崩。

    这个测试存在的原因：CLI 的参数形态（第一个位置参数是预测、
    其余是结果）曾经和文档写的不一致，导致照文档跑会报错。
    """
    import compare_experiments as ce

    def mk_log(n_ctx: int, waves: int, preempt: bool = False) -> str:
        lines, i = [], 1
        if preempt:
            lines.append("INFO Preemption: seq 5 is preempted")
        for _ in range(waves):
            lines.append(
                f"Iteration({i}): {n_ctx} context requests, 4096 context tokens, "
                f"0 generation requests, 0 generation tokens, "
                f"iteration elapsed time: 2.10 ms")
            i += 1
            lines.append(
                f"Iteration({i}): 0 context requests, 0 context tokens, "
                f"{n_ctx} generation requests, 384 generation tokens, "
                f"iteration elapsed time: 40.0 ms")
            i += 1
        return "\n".join(lines)

    res = tmp_path / "results"
    logd = tmp_path / "logs"
    res.mkdir()
    logd.mkdir()
    for tag, tp in (("A", 63.0), ("B", 40.0)):
        (res / f"vllm__{tag}.json").write_text(json.dumps({
            "peak_vram_gib": 7.4,
            "results": [{"workload": "long", "concurrency": 8, "status": "ok",
                         "throughput_tok_s": tp, "ttft_s": 8.0,
                         "success_rate": 1.0}],
        }), encoding="utf-8")
    (logd / "vllm__A.log").write_text(mk_log(2, 4), encoding="utf-8")
    (logd / "vllm__B.log").write_text(mk_log(2, 4, preempt=True), encoding="utf-8")
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps({
        "A": {"concurrency": 2, "throughput": 63.4, "note": "基线"},
        "B": {"concurrency": 2, "throughput": 63.4, "note": "预期无变化"},
    }, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("sys.argv", [
        "compare_experiments.py", str(preds),
        str(res / "vllm__A.json"), str(res / "vllm__B.json"),
        "--log-dir", str(logd),
    ])
    assert ce.main() == 0
    out = capsys.readouterr().out
    assert "| A |" in out and "| B |" in out
    assert "符合" in out and "偏离" in out      # A 符合、B 偏离
    assert "预期无变化" in out
