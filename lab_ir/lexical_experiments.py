"""Reproducible full-corpus positional, compression and TF-IDF experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Callable

from .index import PositionalIndex, build_index, load_index, save_index
from .retrieval import BooleanRetriever, TfidfRetriever, save_skip_table, skip_storage_estimate

COMMON_QUERIES = ["毛巾变黄", "北京旅游攻略", "减脂早餐", "皮肤过敏怎么办", "考研英语"]
BOOLEAN_EXPRESSIONS = [
    "北京 AND (旅游 OR 攻略) AND NOT 租房",
    "(早餐 AND 减脂) OR (鸡蛋 AND 牛奶)",
    "(皮肤 AND 过敏) AND NOT (猫 OR 狗)",
]
SKIP_STEPS = [0, 1, 4, 16, "sqrt", 64]
ORDERS = ["given", "increasing", "decreasing"]


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def samples(index: PositionalIndex, doc_ids: list[int], limit: int = 5) -> list[dict]:
    titles = dict(zip(index.doc_ids, index.titles))
    return [{"doc_id": doc_id, "title": titles[doc_id]} for doc_id in doc_ids[:limit]]


def evaluate(retriever: TfidfRetriever, queries: list[dict],
             positives: dict[str, set[int]], tokenizer: Callable, remove_stopwords: bool) -> dict:
    """Macro means over query requests with at least one known clicked document."""
    begin = time.perf_counter()
    rows, recalls, reciprocal_ranks, query_times = [], [], [], []
    for query in queries:
        started = time.perf_counter()
        tokens = tokenizer(query["query"], remove_stopwords=remove_stopwords)
        results = retriever.search_tokens(tokens, top_k=10)
        query_times.append((time.perf_counter() - started) * 1000)
        top_ids = [r["doc_id"] for r in results]
        relevant = positives.get(query["query_id"], set())
        recall = len(set(top_ids) & relevant) / len(relevant) if relevant else None
        mrr = (next((1 / rank for rank, doc in enumerate(top_ids, 1) if doc in relevant), 0.0)
               if relevant else None)
        if relevant:
            recalls.append(recall)
            reciprocal_ranks.append(mrr)
        rows.append({"query_id": query["query_id"], "query": query["query"],
                     "query_tokens": tokens, "clicked_doc_ids": sorted(relevant),
                     "top_doc_ids": top_ids, "recall_at_10": recall, "mrr_at_10": mrr})
    return {"candidate_count": retriever.index.document_count, "cutoff": 10,
            "total_queries": len(queries), "eligible_queries": len(recalls),
            "excluded_no_clicked_documents": len(queries) - len(recalls),
            "recall_at_10": statistics.mean(recalls) if recalls else None,
            "mrr_at_10": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
            "median_query_ms": statistics.median(query_times) if query_times else None,
            "total_seconds": time.perf_counter() - begin,
            "relevance": "click=1 in search_logs, query_id scoped, fixed corpus only; non-clicks are unjudged",
            "averaging": "macro over eligible query requests; requests with zero known clicks excluded",
            "per_query": rows}


def first100_evaluation(full: dict) -> dict:
    result = dict(full)
    rows = full["per_query"][:100]
    eligible = [r for r in rows if r["recall_at_10"] is not None]
    result.update({"total_queries": len(rows), "eligible_queries": len(eligible),
                   "excluded_no_clicked_documents": len(rows) - len(eligible),
                   "recall_at_10": statistics.mean(r["recall_at_10"] for r in eligible) if eligible else None,
                   "mrr_at_10": statistics.mean(r["mrr_at_10"] for r in eligible) if eligible else None,
                   "selection": "first 100 rows whose split=test, original queries.jsonl order",
                   "per_query": rows})
    # These are full-test timing measurements, not timings of the selected subset.
    result.pop("median_query_ms", None)
    result.pop("total_seconds", None)
    return result


def choose_phrase(index: PositionalIndex, token_path: Path) -> tuple[list[str], list[int], list[int], dict]:
    retriever = BooleanRetriever(index)
    for tokens in (["北京", "旅游"], ["皮肤", "过敏"], ["减脂", "早餐"], ["考研", "英语"]):
        docs, stats = retriever.phrase(tokens)
        if docs:
            return list(tokens), [0, 1], docs, stats
    # A phrase actually occurring within a field, not an invented positive case.
    for record in read_jsonl(token_path):
        for field in ("title", "content"):
            tokens, positions = record[field + "_tokens"], record.get(field + "_positions")
            positions = positions if positions is not None else list(range(len(tokens)))
            for i in range(len(tokens) - 1):
                if len(tokens[i]) > 1 and len(tokens[i + 1]) > 1:
                    pair, offsets = tokens[i:i + 2], positions[i:i + 2]
                    docs, stats = retriever.phrase(pair, offsets)
                    if docs:
                        return pair, offsets, docs, stats
    raise ValueError("The corpus contains no suitable two-token phrase")


def compression_benchmark(index: PositionalIndex, files: dict,
                          phrase_tokens: list[str], phrase_offsets: list[int]) -> dict:
    expected_fingerprint = index.fingerprint()
    reference = BooleanRetriever(index)
    expected_boolean = {expression: reference.search(expression, order="increasing")[0]
                        for expression in BOOLEAN_EXPRESSIONS}
    expected_phrase = reference.phrase(phrase_tokens, phrase_offsets)[0]
    for name, file in files.items():
        print(f"Verifying and timing codec {name}: {Path(file['path']).name}", flush=True)
        started = time.perf_counter()
        decoded = load_index(file["path"])
        load_seconds = time.perf_counter() - started
        retriever = BooleanRetriever(decoded)
        started_search = time.perf_counter()
        first_ids, first_stats = retriever.search(BOOLEAN_EXPRESSIONS[0], order="increasing")
        first_search_seconds = time.perf_counter() - started_search
        cached_times, boolean_correct = [], first_ids == expected_boolean[BOOLEAN_EXPRESSIONS[0]]
        for expression in BOOLEAN_EXPRESSIONS:
            for _ in range(5):
                ids, stats = retriever.search(expression, order="increasing")
                boolean_correct &= ids == expected_boolean[expression]
                cached_times.append(stats["elapsed_ms"])
        phrase_correct = retriever.phrase(phrase_tokens, phrase_offsets)[0] == expected_phrase
        fingerprint = decoded.fingerprint()
        file.update({"load_decode_seconds": load_seconds,
                     "first_boolean_ms_after_decode": first_search_seconds * 1000,
                     "load_plus_first_search_ms": (load_seconds + first_search_seconds) * 1000,
                     "cached_boolean_median_ms": statistics.median(cached_times),
                     "cached_boolean_repeats": len(cached_times),
                     "roundtrip_correct": fingerprint == expected_fingerprint,
                     "boolean_results_correct": bool(boolean_correct),
                     "phrase_results_correct": phrase_correct,
                     "decoded_index_sha256": fingerprint,
                     "timing_scope": "fresh in-process full file read+decode; OS cache uncontrolled; subsequent searches use decoded RAM arrays"})
        if not all((file["roundtrip_correct"], boolean_correct, phrase_correct)):
            raise AssertionError(f"Codec {name} changed the index or retrieval results")
        del retriever, decoded
        gc.collect()
    return files


def run_experiments(dataset_dir: str | Path, outputs_dir: str | Path,
                    tokenizer: Callable | None = None) -> dict:
    if tokenizer is None:
        from .preprocessing import tokenize
        tokenizer = tokenize
    data, output = Path(dataset_dir), Path(outputs_dir)
    output.mkdir(parents=True, exist_ok=True)
    queries = [q for q in read_jsonl(data / "queries.jsonl") if q.get("split") == "test"]
    corpus_ids = {int(doc["doc_id"]) for doc in read_jsonl(data / "documents.jsonl")}
    positives = defaultdict(set)
    for log in read_jsonl(data / "search_logs.jsonl"):
        if log.get("click") == 1 and int(log["doc_id"]) in corpus_ids:
            positives[log["query_id"]].add(int(log["doc_id"]))
    started = time.perf_counter()
    result = {
        "configuration": {
            "dataset_dir": str(data), "document_count": len(corpus_ids),
            "test_request_count": len(queries), "tokenization": "statistical",
            "stopword_modes": ["keep", "drop"], "boolean_precedence": "NOT > AND > OR",
            "phrase_policy": "within title OR within content; never across fields; pre-deletion token offsets",
            "parameter_policy": "TF-IDF and K=10 fixed before test evaluation; no test-based tuning",
            "common_queries": COMMON_QUERIES,
        },
        "indexes": {}, "boolean_examples": [], "phrase_example": {},
        "skip_experiments": [], "skip_tables": [], "order_experiments": [],
        "tfidf": {"weighting": {"tf": "1+ln(tf)", "idf": "ln((N+1)/(df+1))+1",
                                "similarity": "L2-normalized cosine", "fields": "title+content counts combined"}},
        "limitations": [
            "Clicks are observed behavior proxies; unclicked and unexposed documents remain unjudged.",
            "Original train/test has shared users and repeated query texts; this experiment keeps the supplied split.",
            "Compression timings include real decode; RAM search timings exclude decode. OS file cache is not flushed.",
            "Single-level skips benefit conjunctions unevenly; a shorter list or stride can increase comparisons.",
            "Disk comparisons include identical document ID/title metadata; also report dictionary and postings separately.",
        ],
    }
    for mode, remove in (("keep", False), ("drop", True)):
        token_path = output / f"tokens_{mode}.jsonl"
        if not token_path.is_file():
            raise FileNotFoundError(f"Preprocessing output required: {token_path}")
        index = build_index(token_path)
        if set(index.doc_ids) != corpus_ids:
            raise ValueError("Lexical experiments require the complete, fixed document corpus")
        files = save_index(index, output / "indexes", mode)
        if mode == "keep":
            phrase_tokens, phrase_offsets, phrase_docs, phrase_stats = choose_phrase(index, token_path)
            result["phrase_example"] = {
                "tokens": phrase_tokens, "offsets": phrase_offsets, "result_count": len(phrase_docs),
                "sample_results": samples(index, phrase_docs), "stats": phrase_stats,
                "matching_rule": "all query-relative positions equal within one field",
            }
            boolean = BooleanRetriever(index, phrase_tokenizer=lambda text: tokenizer(text, remove_stopwords=False),
                                        term_tokenizer=lambda text: tokenizer(text, remove_stopwords=False))
            and_docs, and_stats = boolean.search(" AND ".join(phrase_tokens), order="increasing")
            phrase_doc_set = set(phrase_docs)
            filtered = [doc for doc in and_docs if doc not in phrase_doc_set]
            false_positives = samples(index, filtered)
            ordinal_lookup = {doc: ordinal for ordinal, doc in enumerate(index.doc_ids)}
            from bisect import bisect_left
            for sample in false_positives:
                sample["positions"] = {}
                for term in phrase_tokens:
                    posting = index.postings[term]
                    location = bisect_left(posting.docs, ordinal_lookup[sample["doc_id"]])
                    sample["positions"][term] = {
                        field: list(posting.positions(location, field)) for field in ("title", "content")}
            result["phrase_example"].update({
                "and_expression": " AND ".join(phrase_tokens), "and_candidate_count": len(and_docs),
                "and_stats": and_stats, "filtered_nonphrase_count": len(filtered),
                "filtered_nonphrase_examples": false_positives,
                "positional_effect": "ordinary AND allows different offsets or different fields; positional matching removes these candidates",
            })
            digest = index.fingerprint()
            for step in SKIP_STEPS:
                result["skip_tables"].append(save_skip_table(
                    index, output / "indexes" / f"keep_skip_{step}.skips", step, digest))
            for expression in BOOLEAN_EXPRESSIONS:
                docs, stats = boolean.search(expression, order="increasing")
                result["boolean_examples"].append({"expression": expression, "result_count": len(docs),
                                                    "sample_results": samples(index, docs), "stats": stats})
            test_expressions = BOOLEAN_EXPRESSIONS + ["的 AND 可以 AND 早餐"]
            for expression in test_expressions:
                reference_docs = None
                for order in ORDERS:
                    measurements = []
                    for _ in range(5):
                        docs, stats = boolean.search(expression, step="sqrt", order=order)
                        measurements.append(stats)
                    if reference_docs is None:
                        reference_docs = docs
                    if docs != reference_docs:
                        raise AssertionError("Reordering changed boolean results")
                    result["order_experiments"].append({
                        "expression": expression, "order": order, "step": "sqrt", **measurements[-1],
                        "elapsed_ms": statistics.median(s["elapsed_ms"] for s in measurements), "repeats": 5,
                        "same_results": True,
                    })
                for step in SKIP_STEPS:
                    measurements = []
                    for _ in range(5):
                        docs, stats = boolean.search(expression, step=step, order="given")
                        measurements.append(stats)
                    if docs != reference_docs:
                        raise AssertionError("Skip pointers changed boolean results")
                    storage = skip_storage_estimate(index, step)
                    result["skip_experiments"].append({
                        "expression": expression, "order": "given", "step": step, **measurements[-1],
                        "elapsed_ms": statistics.median(s["elapsed_ms"] for s in measurements), "repeats": 5,
                        "estimated_pointer_bytes": storage["estimated_bytes"],
                        "index_skip_pointer_count": storage["pointer_count"], "same_results": True,
                        "pointer_storage_scope": storage["representation"],
                        "serialized_skip_file_bytes": next(table["serialized_file_bytes"]
                                                           for table in result["skip_tables"] if table["step"] == step),
                    })
            compression = compression_benchmark(index, files, phrase_tokens, phrase_offsets)
        else:
            # Use terms whose retained offsets may contain gaps in this mode too.
            drop_tokens, drop_offsets, _, _ = choose_phrase(index, token_path)
            compression = compression_benchmark(index, files, drop_tokens, drop_offsets)
        result["indexes"][mode] = {"statistics": index.statistics(), "metadata": index.metadata,
                                    "compression": compression}
        tfidf = TfidfRetriever(index)
        examples = []
        for query in COMMON_QUERIES:
            tokens = tokenizer(query, remove_stopwords=remove)
            examples.append({"query": query, "tokens": tokens, "results": tfidf.search_tokens(tokens, 5)})
        evaluation = evaluate(tfidf, queries, positives, tokenizer, remove)
        result["tfidf"][mode] = {"norm_build_seconds": tfidf.build_seconds,
                                  "examples": examples, "evaluation_test": evaluation,
                                  "evaluation_test100": first100_evaluation(evaluation)}
        print(f"TF-IDF {mode}: Recall@10={evaluation['recall_at_10']:.6f}, MRR@10={evaluation['mrr_at_10']:.6f}", flush=True)
        del tfidf, index
        gc.collect()
    all_samples = result["phrase_example"]["sample_results"] + result["phrase_example"]["filtered_nonphrase_examples"] + [
        sample for example in result["boolean_examples"] for sample in example["sample_results"]] + [
        sample for mode in ("keep", "drop") for example in result["tfidf"][mode]["examples"]
        for sample in example["results"]]
    wanted = {sample["doc_id"] for sample in all_samples}
    excerpts = {doc["doc_id"]: (doc.get("content") or "")[:250]
                for doc in read_jsonl(data / "documents.jsonl") if doc["doc_id"] in wanted}
    for sample in all_samples:
        sample["excerpt"] = excerpts.get(sample["doc_id"], "")
    result["total_seconds"] = time.perf_counter() - started
    (output / "lexical_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Qilin-IR-50K"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    results = run_experiments(args.data, args.output)
    print(json.dumps({"result_file": str(args.output / "lexical_results.json"),
                      "total_seconds": results["total_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
