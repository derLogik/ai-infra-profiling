#!/usr/bin/env python3
"""Parallel HTTP range downloader for HuggingFace files.

Why this exists: `hf download` stalled at 0 bytes on the big safetensors shards
behind the local proxy. Plain curl ranged requests to hf-mirror work fine, so we
fetch fixed-size chunks in parallel, write each to its own part file, and only
concatenate once every chunk of a file has landed. Partial progress survives a
restart: existing complete part files are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CHUNK = 32 * 1024 * 1024


def probe_size(url: str, timeout: int = 60) -> int | None:
    """Ask for byte 0-0 and read the total size out of Content-Range."""
    out = subprocess.run(
        ["curl", "-sL", "-r", "0-0", "-D", "-", "-o", "/dev/null",
         "--max-time", str(timeout), url],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("content-range") and "/" in line:
            tail = line.rsplit("/", 1)[1].strip()
            if tail.isdigit():
                return int(tail)
    return None


def fetch_chunk(url: str, dest: Path, start: int, end: int, timeout: int) -> int:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    cmd = ["curl", "-sL", "--fail", "-r", f"{start}-{end}",
           "--max-time", str(timeout), "-o", str(tmp), url]
    # 代理偶发 SSL 断连（curl rc=35），单块重试几次比整个文件重下划算
    rc = -1
    for attempt in range(4):
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            break
        tmp.unlink(missing_ok=True)
        time.sleep(2 * (attempt + 1))
    if rc != 0:
        raise RuntimeError(f"curl rc={rc} for bytes {start}-{end}")
    size = tmp.stat().st_size
    tmp.replace(dest)
    return size


def download_file(url: str, dest: Path, workers: int, timeout: int) -> None:
    size = probe_size(url)
    if size is None:
        raise SystemExit(f"cannot probe size: {url}")
    need = (size + CHUNK - 1) // CHUNK
    parts = dest.parent / f".parts-{dest.name}"
    parts.mkdir(exist_ok=True)

    todo = []
    for i in range(need):
        start = i * CHUNK
        end = min(start + CHUNK, size) - 1
        part = parts / f"{i:05d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            continue
        todo.append((i, start, end, part))

    print(f"{dest.name}: {size / 2**30:.2f} GiB, {need} chunks, {len(todo)} to fetch", flush=True)
    if not todo:
        print("  already complete, skip", flush=True)

    have = (need - len(todo)) * CHUNK
    started = time.time()
    fetched = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_chunk, url, part, s, e, timeout): (i, part)
                   for i, s, e, part in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            got = fut.result()
            fetched += got
            elapsed = max(time.time() - started, 1e-6)
            rate = fetched / elapsed / 2**20
            pct = (have + fetched) / size * 100
            print(f"  {n}/{len(todo)} chunks {pct:5.1f}%  {rate:5.1f} MB/s", flush=True)

    joined = dest.with_suffix(dest.suffix + ".joining")
    with joined.open("wb") as fh:
        for i in range(need):
            with (parts / f"{i:05d}.part").open("rb") as src:
                while True:
                    buf = src.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    fh.write(buf)
    if joined.stat().st_size != size:
        raise SystemExit(f"size mismatch: {joined.stat().st_size} != {size}")
    joined.replace(dest)
    for p in parts.glob("*.part"):
        p.unlink()
    parts.rmdir()
    print(f"  done {dest} ({size / 2**30:.2f} GiB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="resolve URL prefix ending with repo id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.files:
        download_file(f"{args.base.rstrip('/')}/{name}", out / name,
                      args.workers, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
