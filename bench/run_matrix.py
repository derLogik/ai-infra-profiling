#!/usr/bin/env python3
"""Drive the whole profiling matrix: start each server, bench it, stop it.

Written in Python rather than bash on purpose -- `wsl.exe -- bash -c '...'`
mangles $variables through the Windows shell, which silently broke earlier
attempts. Here every command is an argv list, so nothing gets re-expanded.

For each backend we record:
  - startup_s       : process start -> /health OK (model load + CUDA graph capture)
  - vram_gib        : GPU memory used right after load
  - peak_vram_gib   : max observed while the benchmark was running
  - version         : backend package version, probed from its own venv
  - the client_bench.py result JSON (which now keeps per-request records)

路径不再写死：venv 根目录默认 ~/venvs，可用 --venv-root 覆盖，
也可用 --vllm-bin / --vllm-python / --sglang-python 单独指定。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent


def no_proxy_env() -> dict:
    """Env with all proxy vars stripped.

    这台机器的 no_proxy 写的是 `127.*`，但 curl / urllib 只做字面后缀匹配，
    不认识这个通配符，于是打到 127.0.0.1:8000 的请求被 7892 端口的代理接走
    并返回 502。基准客户端必须显式绕开代理。
    """
    env = os.environ.copy()
    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(k, None)
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    return env


def server_cmd(backend: str, args) -> list[str]:
    if backend == "vllm":
        return [
            args.vllm_bin, "serve", args.model_path,
            "--served-model-name", args.served_name,
            "--port", str(args.port),
            "--host", "127.0.0.1",
            "--quantization", "awq_marlin",
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_mem_util),
            "--max-num-seqs", str(args.max_num_seqs),
            # 所有请求共用同一个 prompt，前缀缓存会让 prefill 全部命中，
            # TTFT 从几百 ms 掉到 30 ms，测的就不是 prefill 成本了。
            *(["--no-enable-prefix-caching"] if args.cold_prefill else []),
            # 实验用透传：例如 --server-arg=--no-scheduler-reserve-full-isl
            # 或 --server-arg=--max-num-batched-tokens --server-arg=512
            *getattr(args, "server_arg", []),
            # 逐 iteration 的并发明细。没有它 compare_experiments 的
            # "实测并发"列会是 —，因为只能从日志反推真实并发结构。
            *(["--enable-logging-iteration-details"]
              if getattr(args, "iteration_details", False) else []),
        ]
    if backend == "sglang":
        return [
            args.sglang_python, "-m", "sglang.launch_server",
            "--model-path", args.model_path,
            "--served-model-name", args.served_name,
            "--port", str(args.port),
            "--host", "127.0.0.1",
            "--quantization", "awq_marlin",
            "--context-length", str(args.max_model_len),
            "--mem-fraction-static", str(args.mem_frac_static),
            "--max-running-requests", str(args.max_num_seqs),
            "--log-level", "warning",
            *(["--disable-radix-cache"] if args.cold_prefill else []),
            # 42 张 CUDA graph 在 8 GB 卡上吃掉近 1 GiB，KV 池被压到
            # 1035 token，2048 token 的 long 负载直接 400。跑 profiling 时
            # 这个交换是划算的：数据完整性 > 解码峰值。
            *(["--disable-cuda-graph"] if not args.sglang_cuda_graph else []),
            *getattr(args, "server_arg", []),
        ]
    raise SystemExit(f"unknown backend: {backend}")


def bench_python(backend: str, args) -> str:
    return args.vllm_python if backend == "vllm" else args.sglang_python


def run_name(backend: str, args) -> str:
    """结果/日志文件基名。带 --tag 时区分同一后端的多组实验。

    标签只做保守清洗：只保留字母数字与 ._-，其余换成 _，
    避免 Windows 上的非法文件名。
    """
    tag = getattr(args, "tag", None)
    if not tag:
        return backend
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(tag))
    return f"{backend}__{safe}"


def probe_version(backend: str, args) -> str:
    """从该后端自己的 venv 里读包版本，避免报告里手写版本号。"""
    pkg = {"vllm": "vllm", "sglang": "sglang"}[backend]
    code = ("import importlib.metadata as m;"
            f"print(m.version('{pkg}'))")
    try:
        out = subprocess.run([bench_python(backend, args), "-c", code],
                             capture_output=True, text=True, timeout=60,
                             env=no_proxy_env())
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class VramSampler:
    """压测期间后台采样显存。

    原实现只在压测结束后取一次，既不是峰值、列名也写错了（peak_vram_gib
    从来没被写入，报告里那一列永远是 `-`）。
    """

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.max = float("nan")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self):
        while not self._stop.is_set():
            v = vram_used()
            if v == v:                     # not NaN
                self.max = v if self.max != self.max else max(self.max, v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_health(port: int, proc: subprocess.Popen, timeout: float) -> float | None:
    url = f"http://127.0.0.1:{port}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if proc.poll() is not None:
            return None
        try:
            with opener.open(url, timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(1.0)
    return None


def server_info(port: int) -> dict:
    """SGLang 的 /get_server_info 给出真实 KV 容量（max_total_num_tokens）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/get_server_info", timeout=5) as r:
            info = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return {}
    keys = ("max_total_num_tokens", "max_req_input_len", "max_context_len",
            "context_length", "max_running_requests")
    return {k: info[k] for k in keys if k in info}


def vram_used() -> float:
    # WSL 的 nvidia-smi 在 /usr/lib/wsl/lib，精简 PATH 下可能找不到；
    # 只是辅助指标，取不到就记 NaN，不要因此中断整轮基准。
    for exe in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi"):
        try:
            out = subprocess.run(
                [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
        except (FileNotFoundError, OSError):
            continue
        try:
            return int(out[0]) / 1024
        except (IndexError, ValueError):
            continue
    return float("nan")


def stop(proc: subprocess.Popen) -> None:
    """SIGTERM only -- never SIGKILL.

    对持有 CUDA 上下文（VLLM::EngineCore）的进程发 SIGKILL，在这台机器的
    WSL 上会把整个 GPU 栈带崩、VM 重启（现象：Wsl/Service/E_UNEXPECTED，
    之后 systemd 又把旧的 vllm-qwen3 服务拉起来抢 8000 端口）。所以这里
    只发 SIGTERM 并耐心等；实在不退就再等，宁可报错也不硬杀。
    """
    if proc.poll() is not None:
        return
    for wait_s in (60, 30):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue
    print(f"  [warn] pid {proc.pid} 未在 90s 内退出，未强杀（避免崩 GPU 栈）",
          flush=True)


def wait_port_free(port: int, timeout: float = 60) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if port_free(port):
            return
        time.sleep(1.0)
    raise SystemExit(f"port {port} still busy after {timeout}s")


def run_backend(backend: str, args) -> dict:
    log_path = args.log_dir / f"{run_name(backend, args)}.log"
    cmd = server_cmd(backend, args)
    print(f"\n=== {backend} ===", flush=True)
    print("  " + " ".join(cmd), flush=True)

    env = no_proxy_env()
    if backend == "vllm":
        # vLLM 0.29 在 WSL 上默认关掉 pinned memory，UvaBuffer 会直接报
        # "UVA is not available"。内核 6.18 满足 >=4.19.121，用官方开关打开。
        env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"

    version = probe_version(backend, args)
    print(f"  version: {version}", flush=True)

    with log_path.open("w", encoding="utf-8") as log:
        t_start = time.perf_counter()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(BENCH_DIR),
                                env=env)
        startup = wait_health(args.port, proc, args.startup_timeout)
        if startup is None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
            stop(proc)
            wait_port_free(args.port)
            raise SystemExit(f"{backend} failed to start:\n{tail}")
        vram_after_load = vram_used()
        print(f"  ready in {startup:.1f}s, VRAM {vram_after_load:.2f} GiB", flush=True)
        info = server_info(args.port)
        if info:
            print(f"  server_info: {info}", flush=True)

        out_path = args.results_dir / f"{run_name(backend, args)}.json"
        bench_cmd = [
            bench_python(backend, args), str(BENCH_DIR / "client_bench.py"),
            "--backend", backend,
            "--base-url", f"http://127.0.0.1:{args.port}",
            "--model", args.served_name,
            "--tokenizer-path", args.model_path,
            "--concurrency", args.concurrency,
            "--repeats", str(args.repeats),
            "--workloads", args.workloads,
            "--max-error-rate", str(args.max_error_rate),
            "--version", version,
            "--out", str(out_path),
        ]
        with VramSampler() as sampler:
            rc = subprocess.run(bench_cmd, env=no_proxy_env()).returncode
        peak = sampler.max
        vram_after_bench = vram_used()
        stop(proc)
        wait_port_free(args.port)

    # 单后端失败不再让整轮矩阵退出：保留已有数据，继续下一个后端。
    if rc != 0:
        print(f"  [warn] {backend} bench exited {rc}，保留已写入的结果", flush=True)

    if not out_path.exists():
        print(f"  [error] {backend} 没有生成 {out_path.name}，跳过", flush=True)
        return {}

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["startup_s"] = round(startup, 2)
    payload["vram_gib"] = round(vram_after_load, 2)
    payload["peak_vram_gib"] = round(peak, 2)
    payload["vram_after_bench_gib"] = round(vram_after_bench, 2)
    payload["version"] = version
    payload["server_args"] = cmd
    payload["server_info"] = info
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = sum(r.get("requests_failed", 0) for r in payload.get("results", []))
    print(f"  done: 失败请求 {failed} 条，peak VRAM {peak:.2f} GiB", flush=True)
    return payload


def main() -> int:
    home = Path(os.path.expanduser("~"))
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--served-name", default="qwen3-8b-awq")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--backends", default="vllm,sglang")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--mem-frac-static", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--sglang-cuda-graph", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="SGLang CUDA graph 默认关闭：8 GB 卡上它能吃掉近 1 GiB KV")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--workloads", default="short,medium,long")
    p.add_argument("--cold-prefill", action=argparse.BooleanOptionalAction, default=True,
                   help="关掉前缀缓存，让每次请求都真正走一遍 prefill")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-error-rate", type=float, default=1.0,
                   help="失败请求占比超过该值则整格标记 error")
    p.add_argument("--startup-timeout", type=float, default=900.0)
    p.add_argument("--results-dir", default=str(BENCH_DIR.parent / "results"))
    p.add_argument("--log-dir", default=str(BENCH_DIR.parent / "logs"))
    # 路径参数化：换机器只需要改这几个，不用动代码
    p.add_argument("--venv-root", default=str(home / "venvs"))
    p.add_argument("--vllm-bin", default=None)
    p.add_argument("--vllm-python", default=None)
    p.add_argument("--sglang-python", default=None)
    # 实验支持：透传任意后端参数 + 给本次运行打标签（结果与日志都带标签，
    # 便于同一个后端跑多组对照，例如 A 基线 / B 关掉 full-ISL 预留）
    p.add_argument("--server-arg", action="append", default=[],
                   metavar="ARG",
                   help="原样追加到后端启动命令，可重复；"
                        "例如 --server-arg=--no-scheduler-reserve-full-isl")
    p.add_argument("--tag", default=None,
                   help="本次运行标签，结果写 {backend}__{tag}.json，日志同理")
    p.add_argument("--iteration-details", action="store_true",
                   help="让后端打印逐 iteration 的并发明细（仅 vLLM）；"
                        "跑准入/并发类实验时必开，否则拿不到实测并发")
    args = p.parse_args()

    venv_root = Path(args.venv_root)
    args.vllm_bin = args.vllm_bin or str(venv_root / "vllm" / "bin" / "vllm")
    args.vllm_python = args.vllm_python or str(venv_root / "vllm" / "bin" / "python")
    args.sglang_python = args.sglang_python or str(venv_root / "sglang" / "bin" / "python")

    args.results_dir = Path(args.results_dir)
    args.log_dir = Path(args.log_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not port_free(args.port):
        raise SystemExit(f"port {args.port} already in use -- 先关掉占用进程")

    for backend in [b.strip() for b in args.backends.split(",") if b.strip()]:
        try:
            run_backend(backend, args)
        except SystemExit as exc:
            print(f"  [error] {backend}: {exc}", flush=True)
            continue
    print("\n全部完成。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
