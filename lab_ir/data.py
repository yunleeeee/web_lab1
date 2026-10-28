"""Reproducible input integrity and descriptive statistics (standard library only)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import statistics


def describe_data(data: Path, output: Path) -> None:
    files = {}
    for path in sorted(data.rglob("*.jsonl")):
        digest, rows = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                rows += 1
        files[str(path.relative_to(data))] = {
            "rows": rows, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
        }
    def read(name):
        with (data / (name + ".jsonl")).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]
    documents, queries, logs = read("documents"), read("queries"), read("search_logs")
    lengths = sorted(len(doc.get("content") or "") for doc in documents)
    ids = {doc["doc_id"] for doc in documents}
    if len(ids) != len(documents):
        raise ValueError("Document IDs must be unique")
    if any(row["doc_id"] not in ids for row in logs):
        raise ValueError("Search logs reference a document outside the supplied corpus")
    profile = {
        "documents": len(documents), "unique_doc_ids": len(ids),
        "content_chars_mean": statistics.mean(lengths),
        "content_chars_median": statistics.median(lengths),
        "content_chars_max": max(lengths), "content_chars_p95": lengths[int(len(lengths) * .95)],
        "queries": len(queries), "query_splits": dict(Counter(q["split"] for q in queries)),
        "search_logs": len(logs), "clicks": sum(row["click"] for row in logs),
        "test_queries_with_clicks": len({row["query_id"] for row in logs if row["split"] == "test" and row["click"]}),
        "empty_title": sum(not (doc.get("title") or "").strip() for doc in documents),
        "empty_content": sum(not (doc.get("content") or "").strip() for doc in documents),
        "duplicate_title_content": len(documents) - len({(doc.get("title"), doc.get("content")) for doc in documents}),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_manifest.json").write_text(json.dumps({
        "data_files": files, "python": platform.python_version(), "platform": platform.platform(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "data_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
