"""Run the four required BGE retrieval experiments on the fixed document corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .preprocessing import STOPWORDS, tokenize
from .semantic import (BGEEncoder, MODEL_REVISION, aggregate_terms, raw_model_text,
                       search, sha256_file)

EXAMPLE_QUERIES = ["毛巾变黄", "北京旅游攻略", "减脂早餐", "皮肤过敏怎么办", "考研英语"]
MODES = ("raw_keep", "raw_drop", "segmented_keep", "segmented_drop")


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fixed_queries(data: Path, corpus_ids: set[int], count: int = 100) -> tuple[list[dict], dict, dict]:
    queries = [row for row in read_jsonl(data / "queries.jsonl") if row.get("split") == "test"][:count]
    ids = {row["query_id"] for row in queries}
    positives = {query_id: set() for query_id in ids}
    outside = 0
    for row in read_jsonl(data / "search_logs.jsonl"):
        if row["query_id"] in ids and row.get("click") == 1:
            if row["doc_id"] in corpus_ids:
                positives[row["query_id"]].add(row["doc_id"])
            else:
                outside += 1
    eligible = sum(bool(positive_ids) for positive_ids in positives.values())
    return queries, positives, {
        "selection": "first 100 split=test rows in queries.jsonl file order",
        "selected_queries": len(queries), "eligible_queries": eligible,
        "excluded_without_corpus_click": len(queries) - eligible,
        "clicked_events_outside_corpus": outside,
        "positive_definition": "search_logs click=1, unique doc IDs within this fixed corpus",
        "limitation": "behavior proxy only; click bias, incomplete exposure and unlabeled documents; unclicked/unexposed does not imply irrelevant",
        "candidate_policy": "rank against the entire fixed corpus, not just logged impressions",
        "query_ids": [row["query_id"] for row in queries],
    }


def evaluate(vectors: np.ndarray, query_vectors: np.ndarray, queries: list[dict],
             positives: dict, doc_ids: np.ndarray) -> dict:
    rows = []
    started = time.perf_counter()
    scores = query_vectors @ vectors.T
    from .semantic import ranked_indices
    for query, query_scores in zip(queries, scores):
        relevant = positives[query["query_id"]]
        if not relevant:
            continue
        ranking = ranked_indices(query_scores, doc_ids, 10)
        ranked_ids = [int(doc_ids[index]) for index in ranking]
        hits = [rank + 1 for rank, doc_id in enumerate(ranked_ids) if doc_id in relevant]
        rows.append({"query_id": query["query_id"], "query": query["query"],
                     "positive_doc_ids": sorted(relevant), "top10_doc_ids": ranked_ids,
                     "recall_at_10": len(hits) / len(relevant),
                     "mrr_at_10": 1 / hits[0] if hits else 0.0})
    return {"eligible_queries": len(rows),
            "recall_at_10": float(np.mean([row["recall_at_10"] for row in rows])) if rows else None,
            "mrr_at_10": float(np.mean([row["mrr_at_10"] for row in rows])) if rows else None,
            "seconds": time.perf_counter() - started, "per_query": rows}


def run(data: Path, output: Path, model_path: Path, device="auto", batch_size=512,
        limit: int | None = None, precision="auto") -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    embeddings = output / "embeddings"
    embeddings.mkdir(exist_ok=True)
    documents = read_jsonl(data / "documents.jsonl", limit)
    doc_ids = np.array([doc["doc_id"] for doc in documents], dtype=np.int64)
    if len(set(doc_ids.tolist())) != len(doc_ids):
        raise ValueError("Duplicate document IDs")
    np.save(embeddings / "doc_ids.npy", doc_ids)
    queries, positives, evaluation_protocol = fixed_queries(data, set(doc_ids.tolist()))
    all_query_texts = [query["query"] for query in queries] + EXAMPLE_QUERIES
    query_tokens = [tokenize(query) for query in all_query_texts]
    token_records = read_jsonl(output / "tokens_keep.jsonl", len(documents))
    if [row["doc_id"] for row in token_records] != doc_ids.tolist():
        raise ValueError("Tokenized input document IDs/order differ from documents.jsonl")
    token_lists = [row["title_tokens"] + row["content_tokens"] for row in token_records]
    vocabulary_list = sorted({term for tokens in token_lists + query_tokens for term in tokens})
    vocabulary = {term: row for row, term in enumerate(vocabulary_list)}
    encoder = BGEEncoder(model_path, device=device, batch_size=batch_size, precision=precision)
    encoder.metadata["model_download"] = json.loads((model_path / "download_metadata.json").read_text())
    run_fingerprint = hashlib.sha256(json.dumps({
        "documents_sha256": sha256_file(data / "documents.jsonl"),
        "tokens_sha256": sha256_file(output / "tokens_keep.jsonl"),
        "limit": limit, "model_revision": MODEL_REVISION, "precision": encoder.precision,
        "stopwords": sorted(STOPWORDS), "raw_numeric_policy": "span-pure-number-><num>",
        "semantic_code_sha256": sha256_file(Path(__file__).with_name("semantic.py")),
        "preprocessing_code_sha256": sha256_file(Path(__file__).with_name("preprocessing.py")),
        "content_tokens_per_chunk": encoder.content_budget,
        "aggregation_policy": "normalized-cls; nonoverlap-token-weighted-text-chunks; occurrence-weighted-independent-terms",
        "vocabulary": vocabulary_list,
    }, ensure_ascii=False).encode()).hexdigest()

    def cached_encode(name: str, texts: list[str]) -> tuple[np.ndarray, dict]:
        path, meta_path = embeddings / f"{name}.npy", embeddings / f"{name}.json"
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("run_fingerprint") == run_fingerprint:
                try:
                    cached = np.load(path, mmap_mode="r")
                    expected_shape = (len(texts), encoder.dim)
                    if (cached.shape == expected_shape and list(cached.shape) == meta.get("shape")
                            and cached.dtype == np.float32 and np.isfinite(cached).all()):
                        print(f"Using complete cache: {name}", flush=True)
                        return cached, meta["statistics"]
                except (OSError, ValueError):
                    pass
        vectors, stats = encoder.encode(texts, name)
        np.save(path, vectors)
        write_json(meta_path, {"run_fingerprint": run_fingerprint, "statistics": stats,
                              "shape": list(vectors.shape), "dtype": str(vectors.dtype)})
        return vectors, stats

    term_vectors, term_stats = cached_encode("term_vectors", vocabulary_list)
    write_json(embeddings / "term_vocabulary.json", vocabulary_list)
    results = {}
    cases = [{"query": query, "results": {}} for query in EXAMPLE_QUERIES]
    raw_texts = [str(doc.get("title") or "") + "\n" + str(doc.get("content") or "") for doc in documents]
    for mode in MODES:
        mode_start = time.perf_counter()
        remove = mode.endswith("drop")
        if mode.startswith("raw"):
            model_texts = [raw_model_text(text, remove) for text in raw_texts]
            vectors, stats = cached_encode(mode, model_texts)
            query_vectors, query_stats = encoder.encode(
                [raw_model_text(text, remove) for text in all_query_texts], mode + "_queries")
        else:
            mode_tokens = [[term for term in tokens if not remove or term not in STOPWORDS]
                           for tokens in token_lists]
            vectors, stats = aggregate_terms(mode_tokens, vocabulary, term_vectors)
            query_vectors, query_stats = aggregate_terms(
                [[term for term in tokens if not remove or term not in STOPWORDS]
                 for tokens in query_tokens], vocabulary, term_vectors)
            np.save(embeddings / f"{mode}.npy", vectors)
            write_json(embeddings / f"{mode}.json", {"run_fingerprint": run_fingerprint,
                       "statistics": stats, "shape": list(vectors.shape), "dtype": str(vectors.dtype)})
        np.save(embeddings / f"{mode}_queries.npy", query_vectors)
        metrics = evaluate(vectors, query_vectors[:len(queries)], queries, positives, doc_ids)
        for case, query_vector in zip(cases, query_vectors[len(queries):]):
            ranking = search(vectors, query_vector, doc_ids, 5)
            case["results"][mode] = [{
                "rank": rank + 1, "doc_id": int(doc_ids[row]), "score": score,
                "title": documents[row].get("title", ""),
                "content_excerpt": (documents[row].get("content") or "")[:550],
            } for rank, (row, score) in enumerate(ranking)]
        results[mode] = {"document_encoding": stats, "query_encoding": query_stats,
                         "embedding_shape": list(vectors.shape), "evaluation": metrics,
                         "mode_seconds": time.perf_counter() - mode_start}
        recall_text = f"{metrics['recall_at_10']:.6f}" if metrics['recall_at_10'] is not None else "N/A"
        mrr_text = f"{metrics['mrr_at_10']:.6f}" if metrics['mrr_at_10'] is not None else "N/A"
        print(f"{mode}: Recall@10={recall_text}; MRR@10={mrr_text}", flush=True)
        write_json(output / "semantic_progress.json", {"completed_modes": list(results),
                                                      "results": results, "cases": cases})
    summary = {
        "document_count": len(documents), "full_corpus": limit is None,
        "model": encoder.metadata, "run_fingerprint": run_fingerprint,
        "field_policy": "title + newline + complete content, with every chunk represented",
        "raw_policy": "NFKC/lowercase/HTML/[话题] cleanup, numeric surfaces -> <num>; preserve punctuation/order; no inserted word separators",
        "stopword_policy": "same resources/stopwords.txt; raw deletes exact Jieba surface spans, replaces each with one space; segmented excludes term embeddings before averaging",
        "segmentation_policy": "Jieba statistical tokenization; each term encoded independently; repeated terms weighted by occurrence count",
        "comparison_caveat": "raw vs segmented also differs in contextual encoding and punctuation retention; this experiment alone cannot attribute every difference exclusively to segmentation",
        "similarity": "cosine via inner product of L2-normalized vectors; document-ID ascending tie break",
        "term_cache": {"unique_terms": len(vocabulary), "encoding": term_stats},
        "evaluation_protocol": evaluation_protocol, "modes": results, "cases": cases,
        "total_seconds": time.perf_counter() - started,
    }
    write_json(output / "semantic_results.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Qilin-IR-50K"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, default=Path("models/bge-small-zh-v1.5"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--precision", choices=["auto", "float32", "float16"], default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="optional smoke test; omit for the required full corpus")
    args = parser.parse_args()
    results = run(args.data, args.output, args.model_path, args.device, args.batch_size,
                  args.limit, args.precision)
    print(json.dumps({"document_count": results["document_count"],
                      "total_seconds": results["total_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
