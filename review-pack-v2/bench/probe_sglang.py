#!/usr/bin/env python3
"""Probe SGLang KV capacity for several (context-length, mem-fraction) pairs.

Why: on this 8 GB card the AWQ weights take 5.68 GiB, and SGLang only exposed
max_total_num_tokens=1041 at --context-length 4096 --mem-fraction-static 0.85,
so a 2048-token prompt is rejected with BAD_REQUEST. Before running the real
matrix we need to know which knob actually buys KV capacity.

Each probe: start server -> /health -> /get_server_info -> SIGTERM.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from pathlib import Path

# 默认值从当前用户家目录推导，换机器不用改代码；
# 仍可用 --python / --model-path 或环境变量覆盖。
HOME = Path.home()
DEFAULT_PY = os.environ.get("SGLANG_PY", str(HOME / "venvs/sglang/bin/python"))
DEFAULT_MODEL = os.environ.get("MODEL_PATH", str(HOME / "models/Qwen3-8B-AWQ"))
PORT = 8000

CONFIGS = [
    (4096, 0.85),
    (4096, 0.90),
    (2560, 0.90),
    (2560, 0.85),
    (3072, 0.88),
]


def env():
    e = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        e.pop(k, None)
    e["NO_PROXY"] = e["no_proxy"] = "127.0.0.1,localhost"
    return e


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_health(proc, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return None
        try:
            with opener().open(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    return None


def info():
    try:
        with opener().open(f"http://127.0.0.1:{PORT}/get_server_info", timeout=5) as r:
            d = json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    return {k: d.get(k) for k in ("max_total_num_tokens", "max_req_input_len",
                                  "context_length", "max_running_requests")}


def build_record(ctx, mem, ok, startup_s=None, info_dict=None,
                 error=None, log_path=None, cmd=None):
    """把一次探测的结果整理成可落盘的一条记录（纯函数，便于单测）。

    为什么要有这个：这个脚本以前只把服务日志写到 /tmp，跑完什么都不留。
    结果 README 里那张标定表的数字没有原始记录可核（见 README §7 的披露）。
    现在每次探测都产出这样一条记录，攒起来就是可复核的数据。
    """
    d = dict(info_dict or {})
    return {
        "context_length_arg": ctx,
        "mem_fraction_static": mem,
        "ok": bool(ok),
        "startup_s": round(startup_s, 2) if startup_s is not None else None,
        # 下面这几个直接来自 /get_server_info，是判断"KV 够不够"的依据
        "max_total_num_tokens": d.get("max_total_num_tokens"),
        "max_req_input_len": d.get("max_req_input_len"),
        "reported_context_length": d.get("context_length"),
        "max_running_requests": d.get("max_running_requests"),
        "error": error,
        "log_path": log_path,
        "cmd": list(cmd) if cmd else None,
    }


def append_record(path, record):
    """读-改-写地追加一条记录。文件不存在就新建。

    每档跑完立刻落盘：SGLang 在显存紧的时候会把 WSL 打崩，
    攒到最后一次性写等于赌它不崩。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if p.exists():
        try:
            records = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except (ValueError, OSError):
            records = []
    records.append(record)
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return len(records)


def run_one(ctx, mem, py=DEFAULT_PY, model=DEFAULT_MODEL):
    cmd = [py, "-m", "sglang.launch_server", "--model-path", model,
           "--served-model-name", "qwen3-8b-awq", "--port", str(PORT),
           "--host", "127.0.0.1", "--quantization", "awq_marlin",
           "--context-length", str(ctx), "--mem-fraction-static", str(mem),
           "--max-running-requests", "8", "--log-level", "warning",
           "--disable-radix-cache", "--disable-cuda-graph"]
    log_path = f"/tmp/probe_{ctx}_{mem}.log"
    log = open(log_path, "w")
    print(f"\n>>> ctx={ctx} mem={mem}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env())
    startup = wait_health(proc)
    got = {}
    error = None
    if startup is None:
        error = f"failed to start (rc={proc.poll()})"
        print(f"    FAILED to start (rc={proc.poll()})", flush=True)
        tail = open(log_path, encoding="utf-8",
                    errors="replace").read()[-600:]
        print("    " + tail.replace("\n", "\n    ")[-600:], flush=True)
    else:
        got = info()
        print(f"    ready in {startup:.0f}s  {got}", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        print("    [warn] did not exit in 90s", flush=True)
    log.close()
    time.sleep(8)
    return build_record(ctx, mem, startup is not None, startup, got,
                        error=error, log_path=log_path, cmd=cmd)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="探测 SGLang 在不同 (context-length, mem-fraction) 下的 KV 容量")
    ap.add_argument("--python", default=DEFAULT_PY,
                    help="SGLang 环境的 python（默认 ~/venvs/sglang/bin/python）")
    ap.add_argument("--model-path", default=DEFAULT_MODEL,
                    help="模型目录（默认 ~/models/Qwen3-8B-AWQ）")
    ap.add_argument("--out", default=None,
                    help="结果落盘路径（默认 results/probe_sglang.json）；"
                         "每跑完一档就写一次，不攒到最后")
    args = ap.parse_args()

    out = args.out or str(
        Path(__file__).resolve().parent.parent / "results" / "probe_sglang.json")
    print(f"结果将写入 {out}", flush=True)
    for ctx, mem in CONFIGS:
        rec = run_one(ctx, mem, py=args.python, model=args.model_path)
        n = append_record(out, rec)
        print(f"    [已落盘 {n}/{len(CONFIGS)}] max_total_num_tokens="
              f"{rec.get('max_total_num_tokens')}", flush=True)
    print("\nprobe done", flush=True)
