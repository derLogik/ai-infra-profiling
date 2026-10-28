#!/usr/bin/env python3
"""probe_sglang 的记录与落盘逻辑单测（纯 CPU，不启动 SGLang）。

为什么值得测：这个脚本以前跑完什么都不留，导致 README 里那张标定表
没有任何原始记录可核（我在 README §7 里加了披露）。现在它把每次探测
整理成结构化记录并每档落盘——这两段逻辑必须有测试，否则"以后就有数据了"
只是一句承诺。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from probe_sglang import append_record, build_record

ROOT = Path(__file__).resolve().parent


@pytest.fixture()
def tmp_path():
    """沙箱禁止用系统临时目录，改用项目内目录（与其他测试文件一致）。"""
    d = ROOT / ".testtmp" / "probe"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_build_record_keeps_info_fields():
    info = {
        "max_total_num_tokens": 1041,
        "max_req_input_len": 1035,
        "context_length": 4096,
        "max_running_requests": 8,
        "something_else": "should be dropped",
    }
    rec = build_record(4096, 0.85, True, 42.345, info,
                       log_path="/tmp/probe_4096_0.85.log", cmd=["python", "-m"])
    assert rec["max_total_num_tokens"] == 1041
    assert rec["max_req_input_len"] == 1035
    assert rec["reported_context_length"] == 4096
    assert rec["max_running_requests"] == 8
    assert "something_else" not in rec
    assert rec["startup_s"] == 42.34 or rec["startup_s"] == 42.35
    assert rec["ok"] is True
    assert rec["context_length_arg"] == 4096
    assert rec["mem_fraction_static"] == 0.85
    assert rec["cmd"] == ["python", "-m"]


def test_build_record_failure_has_error_and_nulls():
    rec = build_record(2048, 0.9, False, None, {}, error="rc=1")
    assert rec["ok"] is False
    assert rec["startup_s"] is None
    assert rec["error"] == "rc=1"
    assert rec["max_total_num_tokens"] is None


def test_build_record_is_json_serializable():
    rec = build_record(4096, 0.88, True, 30.0, {"max_total_num_tokens": 2554})
    assert json.loads(json.dumps(rec))["max_total_num_tokens"] == 2554


def test_append_record_creates_file(tmp_path):
    p = tmp_path / "probe.json"
    n = append_record(p, {"a": 1})
    assert n == 1 and json.loads(p.read_text(encoding="utf-8")) == [{"a": 1}]


def test_append_record_preserves_order(tmp_path):
    p = tmp_path / "probe.json"
    for i, mem in enumerate([0.85, 0.88, 0.90], 1):
        assert append_record(p, {"mem": mem}) == i
    data = json.loads(p.read_text(encoding="utf-8"))
    assert [d["mem"] for d in data] == [0.85, 0.88, 0.90]


def test_append_record_survives_corrupt_file(tmp_path):
    """文件被写坏时不要抛异常——探测本身可能已经把机器打崩过，
    这里再崩一次就彻底丢了数据。"""
    p = tmp_path / "probe.json"
    p.write_text("{not valid json", encoding="utf-8")
    n = append_record(p, {"recovered": True})
    assert n == 1
    assert json.loads(p.read_text(encoding="utf-8"))[0]["recovered"] is True


def test_append_record_creates_parent_dirs(tmp_path):
    p = tmp_path / "results" / "nested" / "probe.json"
    append_record(p, {"x": 1})
    assert p.exists()


def test_append_record_handles_non_list_json(tmp_path):
    """如果文件里是 dict（比如被手工改过），应重置为列表而不是崩。"""
    p = tmp_path / "probe.json"
    p.write_text('{"oops": true}', encoding="utf-8")
    assert append_record(p, {"x": 2}) == 1
    assert json.loads(p.read_text(encoding="utf-8")) == [{"x": 2}]
