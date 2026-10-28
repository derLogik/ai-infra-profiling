#!/usr/bin/env python3
"""把结果文件里的本机绝对路径换成 <HOME> 占位符，避免公开仓库泄露用户名。

只改元数据，不改任何测量值。具体说，它只会动这些地方：
    "server_args": ["/home/<user>/venvs/vllm/bin/vllm", ...]
    "model": "/home/<user>/models/Qwen3-8B-AWQ"
会被换成：
    "<HOME>/venvs/vllm/bin/vllm"
    "<HOME>/models/Qwen3-8B-AWQ"

保留 <HOME> 之后的路径，是因为它仍有信息量（venvs 下哪个环境、模型目录名），
只把用户名这个纯隐私部分去掉。

默认 **dry-run**，只报告将要改什么；加 --in-place 才写回（先写 .bak）。

用法：
    python bench/sanitize_results.py results/vllm.json          # 看看会改什么
    python bench/sanitize_results.py results/*.json --in-place  # 真的改
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Linux/macOS 家目录：/home/alice/... 或 /Users/alice/...
POSIX_HOME = re.compile(r"(?:/home|/Users)/[^/\s\"]+/")
# Windows 家目录：C:\Users\alice\...
WIN_HOME = re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"]+\\")


def sanitize_text(text: str) -> tuple[str, int]:
    """返回 (清洗后文本, 替换处数)。纯函数，便于单测。"""
    text, n1 = POSIX_HOME.subn("<HOME>/", text)
    text, n2 = WIN_HOME.subn(r"<HOME>\\", text)
    return text, n1 + n2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--in-place", action="store_true",
                   help="写回原文件（先备份为 .bak）；不加则只报告")
    args = p.parse_args()

    total = 0
    for raw in args.paths:
        path = Path(raw)
        if not path.exists():
            print(f"跳过（不存在）: {path}")
            continue
        # utf-8-sig：结果文件可能被 PowerShell/记事本写过，带 BOM
        original = path.read_text(encoding="utf-8-sig")
        cleaned, n = sanitize_text(original)
        total += n
        if n == 0:
            print(f"{path.name}: 无需修改")
            continue
        print(f"{path.name}: {n} 处路径将被占位符替换")
        if args.in_place:
            path.with_suffix(path.suffix + ".bak").write_text(
                original, encoding="utf-8")
            path.write_text(cleaned, encoding="utf-8")
            print(f"  已写回（原文件备份为 {path.name}.bak）")

    if total and not args.in_place:
        print(f"\n共 {total} 处。确认无误后加 --in-place 执行。")
    elif total:
        print(f"\n共替换 {total} 处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
