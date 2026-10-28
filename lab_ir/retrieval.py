"""Boolean, field-local phrase, and logarithmic TF-IDF cosine retrieval."""
from __future__ import annotations

import argparse
from array import array
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import heapq
import html
import json
import math
from pathlib import Path
import re
import struct
import time
import unicodedata
from typing import Callable, Sequence

from .index import PositionalIndex, load_index


@dataclass
class SearchStats:
    comparisons: int = 0
    skip_comparisons: int = 0
    skip_jumps: int = 0
    skipped_postings: int = 0
    posting_visits: int = 0
    position_checks: int = 0
    skip_pointer_bytes_built: int = 0
    skip_build_seconds: float = 0.0
    operations: int = 0
    elapsed_ms: float = 0.0
    result_count: int = 0


def skip_step(length: int, step: int | str) -> int:
    if step == "sqrt":
        return max(1, math.isqrt(length))
    value = int(step)
    if value < 0:
        raise ValueError("Skip step cannot be negative; 0 disables skip pointers")
    return value


def build_skips(length: int, step: int | str) -> dict[int, int]:
    """A single level: anchors 0, s, 2s ... point forward exactly s entries."""
    stride = skip_step(length, step)
    if not stride:
        return {}
    return {i: i + stride for i in range(0, max(0, length - stride), stride)}


def skip_storage_estimate(index: PositionalIndex, step: int | str) -> dict:
    pointer_count = 0
    for posting in index.postings.values():
        stride = skip_step(len(posting.docs), step)
        if stride:
            pointer_count += max(0, (len(posting.docs) - 1) // stride)
    return {"step": step, "pointer_count": pointer_count,
            "estimated_bytes": pointer_count * 8,
            "representation": "two uint32 posting offsets (source,target); excludes Python object overhead",
            "posting_order": "dense ordinal ascending; each term has its own sqrt step"}


def save_skip_table(index: PositionalIndex, path: str | Path, step: int | str,
                    index_sha256: str | None = None) -> dict:
    """Persist the same single-layer pointers used by merge intersections.

    Each sorted term has its UTF-8 name, posting length, pointer count, then
    (source,target) pairs as uint32 offsets into that term's posting array.
    Runtime merging builds this table per operand, including intermediate lists.
    """
    started = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"step": step, "index_sha256": index_sha256 or index.fingerprint(),
                         "terms": len(index.postings), "endianness": "little"},
                        separators=(",", ":")).encode("utf-8")
    count = 0
    with path.open("wb") as sink:
        sink.write(b"LIRSKIP1\n")
        sink.write(struct.pack("<I", len(header)))
        sink.write(header)
        for term in sorted(index.postings):
            docs = index.postings[term].docs
            pointers = build_skips(len(docs), step)
            count += len(pointers)
            term_bytes = term.encode("utf-8")
            sink.write(struct.pack("<I", len(term_bytes)))
            sink.write(term_bytes)
            sink.write(struct.pack("<II", len(docs), len(pointers)))
            packed = array("I", (value for pair in pointers.items() for value in pair))
            import sys
            if sys.byteorder != "little":
                packed.byteswap()
            sink.write(packed.tobytes())
    return {"path": str(path), "step": step, "pointer_count": count,
            "pointer_payload_bytes": count * 8, "serialized_file_bytes": path.stat().st_size,
            "dictionary_and_metadata_bytes": path.stat().st_size - count * 8,
            "save_seconds": time.perf_counter() - started,
            "representation": "actual little-endian uint32 source,target pairs, term-local posting offsets",
            "runtime_policy": "merge intersections materialize equivalent per-operand tables; timings include that work"}


def intersection(a: Sequence[int], b: Sequence[int], step: int | str,
                 stats: SearchStats) -> list[int]:
    begin = time.perf_counter()
    a_skips, b_skips = build_skips(len(a), step), build_skips(len(b), step)
    stats.skip_build_seconds += time.perf_counter() - begin
    stats.skip_pointer_bytes_built += 8 * (len(a_skips) + len(b_skips))
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        stats.comparisons += 1
        stats.posting_visits += 2
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            jumped = False
            while i in a_skips:
                target = a_skips[i]
                stats.comparisons += 1
                stats.skip_comparisons += 1
                if a[target] > b[j]:
                    break
                stats.skip_jumps += 1
                stats.skipped_postings += target - i - 1
                i = target
                jumped = True
            if not jumped:
                i += 1
        else:
            jumped = False
            while j in b_skips:
                target = b_skips[j]
                stats.comparisons += 1
                stats.skip_comparisons += 1
                if b[target] > a[i]:
                    break
                stats.skip_jumps += 1
                stats.skipped_postings += target - j - 1
                j = target
                jumped = True
            if not jumped:
                j += 1
    stats.operations += 1
    return result


def union(a: Sequence[int], b: Sequence[int], stats: SearchStats) -> list[int]:
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        stats.comparisons += 1
        stats.posting_visits += 2
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    stats.operations += 1
    return result


def complement(a: Sequence[int], count: int, stats: SearchStats) -> list[int]:
    result, j = [], 0
    for ordinal in range(count):
        stats.posting_visits += 1
        if j < len(a):
            stats.comparisons += 1
        if j < len(a) and a[j] == ordinal:
            j += 1
        else:
            result.append(ordinal)
    stats.operations += 1
    return result


# The grammar is OR(AND(NOT(atom))); quoted strings are positional phrases.
LEXER = re.compile(r'\s*("[^"\n]*"|\(|\)|[^\s()"]+)')


class BooleanSyntaxError(ValueError):
    pass


def parse_boolean(expression: str) -> tuple:
    tokens, position = [], 0
    while position < len(expression):
        if not expression[position:].strip():
            break
        match = LEXER.match(expression, position)
        if not match:
            raise BooleanSyntaxError(f"Invalid boolean syntax at character {position}")
        token = match.group(1)
        tokens.append(token.upper() if token.upper() in ("AND", "OR", "NOT") else token)
        position = match.end()
    current = 0

    def unary():
        nonlocal current
        if current >= len(tokens):
            raise BooleanSyntaxError("Expected a term, phrase or parenthesized expression")
        token = tokens[current]
        current += 1
        if token == "NOT":
            return ("NOT", unary())
        if token == "(":
            node = disjunction()
            if current >= len(tokens) or tokens[current] != ")":
                raise BooleanSyntaxError("Missing closing parenthesis")
            current += 1
            return node
        if token in ("AND", "OR", ")"):
            raise BooleanSyntaxError(f"Unexpected operator: {token}")
        if token.startswith('"'):
            if not token[1:-1].strip():
                raise BooleanSyntaxError("Empty phrase")
            return ("PHRASE", token[1:-1])
        return ("TERM", token)

    def conjunction():
        nonlocal current
        node = unary()
        while current < len(tokens) and tokens[current] == "AND":
            current += 1
            node = ("AND", node, unary())
        return node

    def disjunction():
        nonlocal current
        node = conjunction()
        while current < len(tokens) and tokens[current] == "OR":
            current += 1
            node = ("OR", node, conjunction())
        return node

    if not tokens:
        raise BooleanSyntaxError("Empty boolean expression")
    node = disjunction()
    if current != len(tokens):
        raise BooleanSyntaxError(f"Unexpected token {tokens[current]}; operators must be explicit")
    return node


class BooleanRetriever:
    def __init__(self, index: PositionalIndex,
                 phrase_tokenizer: Callable[[str], list[str]] | None = None,
                 phrase_with_positions: Callable[[str], tuple[list[str], list[int]]] | None = None,
                 term_tokenizer: Callable[[str], list[str]] | None = None):
        self.index = index
        self.phrase_tokenizer = phrase_tokenizer
        self.phrase_with_positions = phrase_with_positions
        self.term_tokenizer = term_tokenizer

    def _term_tokens(self, text: str) -> list[str]:
        if self.term_tokenizer:
            return self.term_tokenizer(text)
        # A bare Boolean atom denotes one indexed lexical item by default.
        # CLI callers configure the corpus tokenizer and expand multiword atoms
        # into conjunctions. Even exact atoms follow corpus surface normalization.
        normalized = unicodedata.normalize("NFKC", html.unescape(text)).lower()
        return ["<num>" if re.fullmatch(r"\d+(?:\.\d+)?", normalized) else normalized]

    def _phrase_tokens(self, text: str) -> tuple[list[str], list[int] | None]:
        if self.phrase_with_positions:
            return self.phrase_with_positions(text)
        if self.phrase_tokenizer:
            return self.phrase_tokenizer(text), None
        from .preprocessing import tokenize_with_positions
        return tokenize_with_positions(text)

    def _phrase_ordinals(self, tokens: Sequence[str], offsets: Sequence[int] | None,
                         step: int | str, stats: SearchStats) -> list[int]:
        if not tokens:
            return []
        if offsets is None:
            offsets = list(range(len(tokens)))
        if len(offsets) != len(tokens) or any(b <= a for a, b in zip(offsets, offsets[1:])):
            raise ValueError("Phrase offsets must strictly increase and align with its tokens")
        relative = [offset - offsets[0] for offset in offsets]
        postings = [self.index.postings.get(token) for token in tokens]
        if any(p is None for p in postings):
            return []
        candidates = postings[0].docs
        # Intersect the rarest lists first, without changing phrase token order.
        for posting in sorted(postings[1:], key=lambda p: len(p.docs)):
            candidates = intersection(candidates, posting.docs, step, stats)
            if not candidates:
                return []
        matches = []
        for ordinal in candidates:
            indices = [bisect_left(posting.docs, ordinal) for posting in postings]
            for field in ("title", "content"):
                positions = [p.positions(i, field) for p, i in zip(postings, indices)]
                if any(not values for values in positions):
                    continue
                following = [set(values) for values in positions[1:]]
                matched = False
                for start in positions[0]:
                    valid = True
                    for offset, values in zip(relative[1:], following):
                        stats.position_checks += 1
                        if start + offset not in values:
                            valid = False
                            break
                    if valid:
                        matched = True
                        break
                if matched:
                    matches.append(ordinal)
                    break
        return matches

    def phrase(self, tokens: Sequence[str], offsets: Sequence[int] | None = None,
               step: int | str = "sqrt") -> tuple[list[int], dict]:
        start, stats = time.perf_counter(), SearchStats()
        matches = self._phrase_ordinals(tokens, offsets, step, stats)
        stats.elapsed_ms = (time.perf_counter() - start) * 1000
        stats.result_count = len(matches)
        return [self.index.doc_ids[i] for i in matches], asdict(stats)

    def search(self, expression: str, step: int | str = "sqrt",
               order: str = "given") -> tuple[list[int], dict]:
        if order not in ("given", "increasing", "decreasing"):
            raise ValueError("Order must be given, increasing, or decreasing")
        start, stats = time.perf_counter(), SearchStats()
        ast = parse_boolean(expression)

        def flatten(node, operator):
            if node[0] == operator:
                return flatten(node[1], operator) + flatten(node[2], operator)
            return [node]

        def evaluate(node):
            op = node[0]
            if op == "TERM":
                tokens = self._term_tokens(node[1])
                if not tokens:
                    return []
                lists = [self.index.postings[token].docs if token in self.index.postings else []
                         for token in tokens]
                if order != "given":
                    lists.sort(key=len, reverse=order == "decreasing")
                result = lists[0]
                for posting in lists[1:]:
                    result = intersection(result, posting, step, stats)
                return result
            if op == "PHRASE":
                tokens, offsets = self._phrase_tokens(node[1])
                return self._phrase_ordinals(tokens, offsets, step, stats)
            if op == "NOT":
                return complement(evaluate(node[1]), self.index.document_count, stats)
            operands = [evaluate(child) for child in flatten(node, op)]
            if order != "given":
                operands.sort(key=len, reverse=order == "decreasing")
            result = operands[0]
            for operand in operands[1:]:
                if op == "AND":
                    if not result:
                        break
                    result = intersection(result, operand, step, stats)
                else:
                    result = union(result, operand, stats)
            return result

        matches = evaluate(ast)
        stats.elapsed_ms = (time.perf_counter() - start) * 1000
        stats.result_count = len(matches)
        return [self.index.doc_ids[i] for i in matches], asdict(stats)


class TfidfRetriever:
    """Title and content term counts combined; weights (1+ln tf)*smoothed idf."""
    def __init__(self, index: PositionalIndex):
        started = time.perf_counter()
        self.index = index
        self.idf = {term: math.log((index.document_count + 1) / (len(p.docs) + 1)) + 1
                    for term, p in index.postings.items()}
        squared = array("d", [0.0]) * index.document_count
        for term, posting in index.postings.items():
            idf = self.idf[term]
            for p, ordinal in enumerate(posting.docs):
                weight = (1 + math.log(posting.term_frequency(p))) * idf
                squared[ordinal] += weight * weight
        self.norms = array("d", (math.sqrt(value) for value in squared))
        self.build_seconds = time.perf_counter() - started

    def search_tokens(self, tokens: Sequence[str], top_k: int = 10) -> list[dict]:
        if top_k < 1:
            return []
        query_tf = Counter(tokens)
        query_weights = {term: (1 + math.log(tf)) * self.idf[term]
                         for term, tf in query_tf.items() if term in self.idf}
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        if not query_norm:
            return []
        scores = defaultdict(float)
        for term, query_weight in query_weights.items():
            posting = self.index.postings[term]
            idf = self.idf[term]
            for p, ordinal in enumerate(posting.docs):
                document_weight = (1 + math.log(posting.term_frequency(p))) * idf
                scores[ordinal] += document_weight * query_weight
        for ordinal in scores:
            scores[ordinal] /= self.norms[ordinal] * query_norm
        ranked = heapq.nlargest(top_k, scores,
                               key=lambda ordinal: (scores[ordinal], -self.index.doc_ids[ordinal]))
        return [{"doc_id": self.index.doc_ids[ordinal], "score": scores[ordinal],
                 "title": self.index.titles[ordinal]} for ordinal in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("outputs/indexes/keep_combined.idx"))
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=("boolean", "phrase", "tfidf"), default="tfidf")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip", default="sqrt")
    parser.add_argument("--order", choices=("given", "increasing", "decreasing"), default="increasing")
    parser.add_argument("--drop-stopwords", action="store_true")
    args = parser.parse_args()
    from .preprocessing import tokenize, tokenize_with_positions
    index = load_index(args.index)
    tokenize_phrase = lambda text: tokenize_with_positions(text, remove_stopwords=args.drop_stopwords)
    if args.mode == "tfidf":
        tokens = tokenize(args.query, remove_stopwords=args.drop_stopwords)
        results = TfidfRetriever(index).search_tokens(tokens, args.top_k)
        print(json.dumps({"query_tokens": tokens, "results": results}, ensure_ascii=False, indent=2))
    else:
        retriever = BooleanRetriever(index, phrase_with_positions=tokenize_phrase,
                                     term_tokenizer=lambda text: tokenize(text, remove_stopwords=args.drop_stopwords))
        if args.mode == "phrase":
            tokens, offsets = tokenize_phrase(args.query)
            docs, stats = retriever.phrase(tokens, offsets, args.skip)
        else:
            docs, stats = retriever.search(args.query, args.skip, args.order)
        titles = dict(zip(index.doc_ids, index.titles))
        print(json.dumps({"total": len(docs), "results": [
            {"doc_id": doc, "title": titles[doc]} for doc in docs[:args.top_k]], "stats": stats},
            ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
