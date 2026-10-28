#!/usr/bin/env python3
"""把多组实验的结果并排比较，并且**强制把预测写在前面**。

为什么需要它：控制变量实验最容易犯的错是"跑完再看结果，然后解释"。
那叫事后合理化，不叫实验。这个脚本要求你先写一份 predictions，
再让实测值贴着预测值显示——对不上就是"偏离预测"，跑不掉。

配套关系：
    run_matrix.py --backends vllm --tag A ...          产生 results/vllm__A.json + logs/vllm__A.log
    run_matrix.py --backends vllm --tag B --server-arg=...  产生 vllm__B.*
    compare_experiments.py predictions.json results/vllm__A.json results/vllm__B.json

predictions 格式（键 = tag，值 = 预测）：

    {
      "A": {"concurrency": 2, "throughput": 63.4, "ttft_median": 8.09,
            "note": "基线：2 路并发，4 波"},
      "B": {"concurrency": 2, "throughput": 63.4, "ttft_median": 8.09,
            "note": "预期无变化：chunk 仍是整个 prompt，照样装不下"}
    }

用法：
    python bench/compare_experiments.py preds.json results/vllm__A.json results/vllm__B.json
    python bench/compare_experiments.py preds.json results/vllm__*.json --workload long --concurrency 8
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

try:
    from analyze_iterations import parse as parse_iterations
    from analyze_iterations import summarize as summarize_iterations
except Exception:  # pragma: no cover - 日志分析缺失时退化为不做并发判读
    parse_iterations = None
    summarize_iterations = None


def tag_of(path: Path) -> str:
    """从 vllm__A.json 取出 A；没有标签时返回 stem。"""
    stem = path.stem
    return stem.split("__", 1)[1] if "__" in stem else stem


def find_cell(payload: dict, workload: str, concurrency: int) -> dict | None:
    for cell in payload.get("results") or []:
        if cell.get("workload") == workload and cell.get("concurrency") == concurrency:
            return cell
    return None


def cell_metrics(payload: dict, workload: str, concurrency: int) -> dict:
    """抽取一格的关键指标；缺数据时全部返回 nan/None，不抛异常。

    这种"缺就标 —"的行为是刻意的：一组实验跑挂了不该让整个对比表消失，
    那样会让人倾向于删掉失败的那组。
    """
    cell = find_cell(payload, workload, concurrency)
    if cell is None:
        return {"present": False}
    return {
        "present": True,
        "status": cell.get("status", "ok"),
        "throughput": cell.get("throughput_tok_s"),
        "ttft_median": cell.get("ttft_s"),
        "ttft_p95": cell.get("req_ttft_s_p95"),
        "success_rate": cell.get("success_rate"),
        "peak_vram": payload.get("peak_vram_gib"),
    }


def count_preemptions(log_text: str) -> int:
    """日志里带 preempt 的行数。

    启发式，不是精确事件计数：用于回答"这组实验里到底有没有发生抢占"。
    比精确数字更重要的是**有没有**——有抢占就说明这条路线在拿重算换并发。
    """
    n = 0
    for line in log_text.splitlines():
        low = line.lower()
        if "preempt" in low and "num_preempt" not in low:
            n += 1
    return n


def iteration_summary(log_path: Path, expect_requests: int) -> dict:
    if not (parse_iterations and summarize_iterations) or not log_path.exists():
        return {"ok": False}
    try:
        steps = parse_iterations(log_path)
        return summarize_iterations(steps, expect_requests)
    except Exception:
        return {"ok": False}


def _fmt(v, digits: int = 1) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "—"
    return f"{f:.{digits}f}"


def verdict(obs, pred, rel_tol: float) -> str:
    """单格判定：实测是否落在预测的容差内。"""
    if obs is None or pred is None:
        return "无法判定"
    try:
        o, p = float(obs), float(pred)
    except (TypeError, ValueError):
        return "无法判定"
    if math.isnan(o) or math.isnan(p):
        return "无法判定"
    if p == 0:
        return "符合" if o == 0 else "偏离"
    return "符合" if abs(o - p) / abs(p) <= rel_tol else "偏离"


def render(arms: list[dict], predictions: dict, rel_tol: float = 0.15) -> str:
    """arms: [{tag, metrics, concurrency_obs, preemptions}, ...]"""
    rows = [
        "| 组 | 预测并发 | 实测并发 | 预测吞吐 | 实测吞吐 | TTFT 中位(预测/实测) | 抢占行数 | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        tag = a["tag"]
        p = predictions.get(tag) or {}
        m = a["metrics"]
        rows.append(
            "| {tag} | {pc} | {oc} | {pt} | {ot} | {pm} / {om} | {pre} | {v} |".format(
                tag=tag,
                pc=_fmt(p.get("concurrency"), 0),
                oc=_fmt(a.get("concurrency_obs"), 0),
                pt=_fmt(p.get("throughput")),
                ot=_fmt(m.get("throughput")),
                pm=_fmt(p.get("ttft_median"), 2),
                om=_fmt(m.get("ttft_median"), 2),
                pre=a.get("preemptions", "—"),
                v=verdict(m.get("throughput"), p.get("throughput"), rel_tol),
            )
        )

    notes = [f"- **{a['tag']}**：{predictions.get(a['tag'], {}).get('note', '（未写预测）')}"
             for a in arms]
    return "\n".join([*rows, "", "预测与预期：", *notes])


def build_arm(json_path: Path, log_path: Path | None, workload: str,
              concurrency: int, expect_requests: int) -> dict:
    # utf-8-sig：Windows 记事本 / PowerShell 5.1 会写 BOM，用 utf-8 读会炸。
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    metrics = cell_metrics(payload, workload, concurrency)
    pre = 0
    observed_conc = None
    if log_path and log_path.exists():
        text = log_path.read_text(encoding="utf-8-sig", errors="replace")
        pre = count_preemptions(text)
        s = iteration_summary(log_path, expect_requests)
        if s.get("ok"):
            observed_conc = s.get("max_concurrency")
    return {"tag": tag_of(json_path), "metrics": metrics,
            "preemptions": pre, "concurrency_obs": observed_conc}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("predictions", help="预测 JSON（键为 tag）")
    p.add_argument("results", nargs="+", help="results/*.json")
    p.add_argument("--workload", default="long")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--expect-requests", type=int, default=8)
    p.add_argument("--log-dir", default=str(BENCH_DIR.parent / "logs"))
    p.add_argument("--rel-tol", type=float, default=0.15,
                   help="吞吐判定容差，默认 15%%")
    args = p.parse_args()

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8-sig"))
    log_dir = Path(args.log_dir)
    arms = []
    for rp in map(Path, args.results):
        lp = log_dir / (rp.stem + ".log")
        arms.append(build_arm(rp, lp, args.workload, args.concurrency,
                              args.expect_requests))
    print(f"工作负载 {args.workload} / 并发 {args.concurrency}\n")
    print(render(arms, predictions, args.rel_tol))

    # 实测并发缺失是最常见的"表看着对但结论没法下"的原因，
    # 与其让人对着 `—` 猜，不如直接说清怎么补。
    missing = [a["tag"] for a in arms if a.get("concurrency_obs") is None]
    if missing:
        print(f"\n[提示] 这些组拿不到实测并发：{', '.join(missing)}")
        print("      原因是日志里没有 iteration 明细。重跑时给 run_matrix.py 加 "
              "--iteration-details，")
        print("      或手动传 --server-arg=--enable-logging-iteration-details。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
