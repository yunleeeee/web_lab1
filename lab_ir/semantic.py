"""Frozen BGE embeddings: complete text chunks and true per-term aggregation.

The segmented experiment encodes every lexical term independently. A document
is the L2-normalized frequency-weighted mean of those normalized vectors. It
does not submit a space-joined segmented sentence to the transformer.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from typing import Sequence

import jieba
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .preprocessing import NUMBER, STOPWORDS, initialize, normalize_text, tokenize

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
MODEL_SHA256 = "354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026"
MODEL_CARD = "https://huggingface.co/BAAI/bge-small-zh-v1.5"


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).tiny)


def raw_model_text(text: str, remove_stopwords: bool = False) -> str:
    """Keep original word order/punctuation, apply shared numeric normalization.

    Jieba identifies spans only to replace numeric surfaces and (optionally)
    stopwords. All other surfaces are concatenated without adding separators;
    a deleted stopword becomes a space so neighboring words cannot be glued.
    """
    initialize()
    surfaces = jieba.lcut(normalize_text(text), HMM=True)
    result = []
    for surface in surfaces:
        lexical = "<num>" if NUMBER.fullmatch(surface.strip()) else surface
        result.append(" " if remove_stopwords and lexical.strip() in STOPWORDS else lexical)
    return "".join(result)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BGEEncoder:
    def __init__(self, model_path: Path, device: str = "auto", batch_size: int = 512,
                 precision: str = "auto", padded_token_budget: int = 16384):
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.precision = "float16" if precision == "auto" and device != "cpu" else (
            "float32" if precision == "auto" else precision)
        self.batch_size = batch_size
        self.padded_token_budget = padded_token_budget
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True,
                                               attn_implementation="sdpa")
        self.model.eval().to(device)
        if self.precision == "float16":
            self.model.half()
        self.dim = self.model.config.hidden_size
        self.max_length = min(self.model.config.max_position_embeddings,
                              self.tokenizer.model_max_length)
        self.content_budget = self.max_length - self.tokenizer.num_special_tokens_to_add(False)
        if self.dim != 512 or self.max_length != 512:
            raise ValueError(f"Unexpected BGE configuration: {self.dim=}, {self.max_length=}")
        weight_path = model_path / "model.safetensors"
        actual_sha = sha256_file(weight_path)
        if actual_sha != MODEL_SHA256:
            raise ValueError("Model weights do not match the frozen revision SHA256")
        self.metadata = {
            "model_id": MODEL_ID, "revision": MODEL_REVISION, "model_card": MODEL_CARD,
            "weights_sha256": actual_sha, "dimension": self.dim,
            "maximum_sequence_length": self.max_length, "content_tokens_per_chunk": self.content_budget,
            "pooling": "normalized CLS per chunk; token-count-weighted chunk mean; L2 normalization",
            "query_instruction": "none; v1.5 supports no instruction; equal policy across all modes",
            "device": device, "precision": self.precision,
            "batch_size_limit": batch_size, "padded_token_budget": padded_token_budget,
            "dependencies": {package: importlib.metadata.version(package)
                             for package in ("torch", "transformers", "numpy", "jieba", "safetensors")},
        }

    def encode(self, texts: Sequence[str], label: str = "texts") -> tuple[np.ndarray, dict]:
        """Encode complete inputs, with no silently discarded model tokens."""
        started = time.perf_counter()
        chunks: list[tuple[int, list[int]]] = []
        token_lengths: list[int] = []
        # Tokenize in blocks to limit temporary tokenizer allocations.
        for start in range(0, len(texts), 1024):
            encoded = self.tokenizer(list(texts[start:start + 1024]), add_special_tokens=False,
                                     truncation=False, padding=False)["input_ids"]
            for offset, ids in enumerate(encoded):
                token_lengths.append(len(ids))
                for part in range(0, max(1, len(ids)), self.content_budget):
                    chunks.append((start + offset, ids[part:part + self.content_budget]))
        # Equal-length batches reduce padding; all chunks keep their original owner.
        chunks.sort(key=lambda item: len(item[1]))
        sums = np.zeros((len(texts), self.dim), dtype=np.float32)
        token_weights = np.zeros(len(texts), dtype=np.float64)
        batch_start = 0
        last_progress = started
        with torch.inference_mode():
            while batch_start < len(chunks):
                stop = min(len(chunks), batch_start + self.batch_size)
                while stop > batch_start + 1 and (
                    (stop - batch_start) * (len(chunks[stop - 1][1]) + 2) > self.padded_token_budget
                ):
                    stop -= 1
                batch = chunks[batch_start:stop]
                prepared = [{"input_ids": [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id],
                             "attention_mask": [1] * (len(ids) + 2)} for _, ids in batch]
                inputs = self.tokenizer.pad(prepared, padding=True, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs).last_hidden_state[:, 0].float()
                vectors = torch.nn.functional.normalize(outputs, p=2, dim=1).cpu().numpy()
                for (owner, ids), vector in zip(batch, vectors):
                    weight = max(1, len(ids))
                    sums[owner] += weight * vector
                    token_weights[owner] += weight
                batch_start = stop
                now = time.perf_counter()
                if now - last_progress >= 20 or batch_start == len(chunks):
                    print(f"{label}: {batch_start:,}/{len(chunks):,} chunks; "
                          f"{now - started:.1f}s", flush=True)
                    last_progress = now
        if len(texts):
            sums /= token_weights[:, None].astype(np.float32)
        statistics = {
            "input_count": len(texts), "model_content_tokens": sum(token_lengths),
            "chunk_count": len(chunks), "multi_chunk_inputs": sum(n > self.content_budget for n in token_lengths),
            "max_input_model_tokens": max(token_lengths, default=0),
            "truncated_inputs": 0, "discarded_model_tokens": 0,
            "empty_inputs": sum(n == 0 for n in token_lengths),
            "seconds": time.perf_counter() - started,
        }
        return normalize_rows(sums), statistics


def aggregate_terms(token_lists: Sequence[Sequence[str]], vocabulary: dict[str, int],
                    term_vectors: np.ndarray) -> tuple[np.ndarray, dict]:
    """Exactly equal to a mean over every independently encoded token occurrence."""
    vectors = np.zeros((len(token_lists), term_vectors.shape[1]), dtype=np.float32)
    empty = occurrences = 0
    for row, tokens in enumerate(token_lists):
        if not tokens:
            empty += 1
            continue
        counts = Counter(tokens)
        indices = np.array([vocabulary[term] for term in counts], dtype=np.int64)
        frequencies = np.array(list(counts.values()), dtype=np.float32)
        vectors[row] = frequencies @ term_vectors[indices] / len(tokens)
        occurrences += len(tokens)
    return normalize_rows(vectors), {
        "input_count": len(token_lists), "token_occurrences": occurrences,
        "empty_inputs": empty, "aggregation": "term-frequency-weighted arithmetic mean of independent normalized term vectors; L2 normalization",
    }


def ranked_indices(scores: np.ndarray, doc_ids: np.ndarray, k: int = 10) -> np.ndarray:
    """Deterministic cosine ranking, with smaller document ID breaking ties."""
    return np.lexsort((doc_ids, -scores))[:k]


def search(vectors: np.ndarray, query_vector: np.ndarray, doc_ids: np.ndarray,
           k: int = 10) -> list[tuple[int, float]]:
    scores = vectors @ query_vector
    return [(int(row), float(scores[row])) for row in ranked_indices(scores, doc_ids, k)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the cached BGE experiment embeddings")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["raw_keep", "raw_drop", "segmented_keep", "segmented_drop"],
                        default="raw_keep")
    parser.add_argument("--data", type=Path, default=Path("Qilin-IR-50K"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--model-path", type=Path, default=Path("models/bge-small-zh-v1.5"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    embedding_dir = args.output / "embeddings"
    vectors = np.load(embedding_dir / f"{args.mode}.npy", mmap_mode="r")
    doc_ids = np.load(embedding_dir / "doc_ids.npy")
    metadata = json.loads((args.output / "semantic_results.json").read_text())
    encoder = BGEEncoder(args.model_path, device=args.device,
                         precision=metadata["model"]["precision"])
    remove = args.mode.endswith("drop")
    if args.mode.startswith("raw"):
        query_vector = encoder.encode([raw_model_text(args.query, remove)], "query")[0][0]
    else:
        terms = tokenize(args.query, remove_stopwords=remove)
        term_vocabulary = json.loads((embedding_dir / "term_vocabulary.json").read_text())
        term_lookup = {term: row for row, term in enumerate(term_vocabulary)}
        term_vectors = np.load(embedding_dir / "term_vectors.npy", mmap_mode="r")
        unknown = sorted(set(terms) - set(term_lookup))
        unknown_vectors = encoder.encode(unknown, "new query terms")[0] if unknown else None
        unknown_lookup = {term: row for row, term in enumerate(unknown)}
        query_vector = np.zeros(encoder.dim, dtype=np.float32)
        for term in terms:
            query_vector += (term_vectors[term_lookup[term]] if term in term_lookup
                             else unknown_vectors[unknown_lookup[term]])
        query_vector = normalize_rows(query_vector[None, :])[0]
    documents = {}
    with (args.data / "documents.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            documents[doc["doc_id"]] = doc
    results = []
    for rank, (row, score) in enumerate(search(vectors, query_vector, doc_ids, args.top_k)):
        doc_id = int(doc_ids[row])
        doc = documents[doc_id]
        results.append({"rank": rank + 1, "doc_id": doc_id, "score": score,
                        "title": doc.get("title", ""),
                        "content_excerpt": (doc.get("content") or "")[:550]})
    print(json.dumps({"query": args.query, "mode": args.mode, "results": results},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
