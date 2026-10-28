#!/usr/bin/env python3
"""Download the frozen BGE model without executing code from the model repo.

Uses only Python's standard library. Parallel HTTP ranges can resume on rerun.
The completed weight file must match the official revision's SHA256 before it
becomes model.safetensors. For restricted networks, pass an explicit endpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

MODEL_ID = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
WEIGHT_SHA256 = "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026"
WEIGHT_SIZE = 95827648
FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "vocab.txt", "README.md", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json"]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(url: str, destination: Path, timeout: int, expected_bytes: int | None = None,
          byte_range: tuple[int, int] | None = None) -> None:
    headers = {"User-Agent": "Lab1-Reproducible-Model-Downloader/1.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(3):
        temporary = destination.with_suffix(destination.suffix + ".downloading")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as target:
                if byte_range:
                    expected_content_range = f"bytes {byte_range[0]}-{byte_range[1]}/{WEIGHT_SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_content_range:
                        raise ValueError("Endpoint did not return the requested byte range")
                shutil.copyfileobj(response, target, length=1024 * 1024)
            if expected_bytes is not None and temporary.stat().st_size != expected_bytes:
                raise ValueError(f"Incomplete download: {destination.name}")
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download(endpoint: str, destination: Path, workers: int, timeout: int) -> None:
    endpoint = endpoint.rstrip("/")
    destination.mkdir(parents=True, exist_ok=True)
    base_url = f"{endpoint}/{MODEL_ID}/resolve/{REVISION}/"
    for filename in FILES:
        path = destination / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch(base_url + filename, path, timeout)
        print(f"Downloaded {filename}: {path.stat().st_size:,} bytes", flush=True)
    weights = destination / "model.safetensors"
    if weights.exists() and weights.stat().st_size == WEIGHT_SIZE and sha256(weights) == WEIGHT_SHA256:
        print("Using existing verified weights", flush=True)
    else:
        part_folder = destination / f"download_parts_{workers}"
        part_folder.mkdir(exist_ok=True)
        step = math.ceil(WEIGHT_SIZE / workers)

        def part(index: int) -> tuple[int, int]:
            start, stop = index * step, min(WEIGHT_SIZE, (index + 1) * step) - 1
            target = part_folder / f"{index:03}.part"
            expected = stop - start + 1
            if not target.exists() or target.stat().st_size != expected:
                fetch(base_url + "model.safetensors", target, timeout,
                      expected_bytes=expected, byte_range=(start, stop))
            return index, expected

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(part, index) for index in range(workers)]
            for future in as_completed(pending):
                index, size = future.result()
                print(f"Verified range size {index + 1}/{workers}: {size:,} bytes", flush=True)
        complete = destination / "model.complete.safetensors"
        with complete.open("wb") as target:
            for index in range(workers):
                with (part_folder / f"{index:03}.part").open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        actual_sha = sha256(complete)
        if actual_sha != WEIGHT_SHA256:
            complete.unlink(missing_ok=True)
            raise ValueError(f"SHA256 mismatch: {actual_sha}; delete {part_folder} and retry")
        complete.replace(weights)
        shutil.rmtree(part_folder)
    metadata = {"model_id": MODEL_ID, "revision": REVISION,
                "official_model_card": f"https://huggingface.co/{MODEL_ID}",
                "download_endpoint": endpoint,
                "weight_sha256": WEIGHT_SHA256, "weight_bytes": WEIGHT_SIZE,
                "verification": "completed weights SHA256 matched frozen official revision"}
    (destination / "download_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Ready: {destination.resolve()}; SHA256={WEIGHT_SHA256}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--destination", type=Path, default=Path("models/bge-small-zh-v1.5"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=150,
                        help="network timeout per range; completed ranges are kept for retry")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    if not args.endpoint.startswith("https://"):
        parser.error("--endpoint must use HTTPS")
    download(args.endpoint, args.destination, args.workers, args.timeout)


if __name__ == "__main__":
    main()
