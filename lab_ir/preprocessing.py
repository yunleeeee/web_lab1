"""Dictionary forward/reverse maximum matching and probabilistic Jieba tokenization.

Positions count normalized lexical tokens before stopword removal. Fields are
kept separate so phrase retrieval cannot cross the title/content boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import time
import unicodedata

import jieba

#准备路径，听用词和匹配规则。
ROOT = Path(__file__).resolve().parents[1]
STOPWORD_PATH = ROOT / "resources" / "stopwords.txt"
STOPWORDS = frozenset(
    line.strip() for line in STOPWORD_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
)
ASCII_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.I)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
LEXICAL = re.compile(r"[\u3400-\u9fffA-Za-z0-9]")
_dictionary: frozenset[str] | None = None
_max_word_length = 0

#统一文本形式。
'''
输入：ＡＢＣ&nbsp;１２３#旅行[话题]#
输出：abc 123#旅行#
'''
def normalize_text(text: str) -> str:
    """NFKC, HTML entities, lowercase and removal of Xiaohongshu topic markup."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text or ""))).lower()
    return re.sub(r"\[话题\]", "", text)

#加载词典
def initialize() -> None:
    global _dictionary, _max_word_length
    if _dictionary is None:
        jieba.initialize()
        _dictionary = frozenset(word for word, freq in jieba.dt.FREQ.items() if freq > 0)
        _max_word_length = max(map(len, _dictionary))

#正向匹配函数
def forward_maximum_matching(text: str) -> list[str]:
    """Pure dictionary matching, with ASCII runs and one-character OOV fallback."""
    initialize()
    result: list[str] = []
    i = 0
    while i < len(text):
        match = ASCII_WORD.match(text, i)
        if match:
            result.append(match.group())
            i = match.end()
            continue
        #跳过标点，空格或表情等无效词。
        if not LEXICAL.match(text[i]):
            i += 1
            continue
        end = min(len(text), i + _max_word_length)
        while end > i + 1 and text[i:end] not in _dictionary:
            end -= 1
        result.append(text[i:end])
        i = end
    return result

#逆向匹配函数
def reverse_maximum_matching(text: str) -> list[str]:
    """Match dictionary words from right to left, keeping ASCII runs readable."""
    initialize()
    result: list[str] = []
    reversed_text = text[::-1]
    end = len(text)

    while end > 0:
        # 从右侧识别英文数字片段，保留片段原来的字符顺序
        match = ASCII_WORD.match(reversed_text, len(text) - end)
        if match:
            piece = match.group()[::-1]
            result.append(piece)
            end -= len(piece)
            continue

        # 跳过标点、空格、表情等字符
        if not LEXICAL.match(text[end - 1]):
            end -= 1
            continue

        # 固定右边界，从最长候选词开始匹配
        start = max(0, end - _max_word_length)
        while start < end - 1 and text[start:end] not in _dictionary:
            start += 1

        result.append(text[start:end])
        end = start

    # 恢复原文中的词项顺序，其实中文本身是没有反转的，但是词语的顺序反了，逆向匹配最先加入的是最后面的词。
    result.reverse()
    return result

#组织完整的单段文本处理流程。
def tokenize_with_positions(
    text: str, method: str = "statistical", remove_stopwords: bool = False,
) -> tuple[list[str], list[int]]:
    normalized = normalize_text(text)
    if method == "matching":
        pieces = forward_maximum_matching(normalized)
    elif method == "reverse_matching":
        pieces = reverse_maximum_matching(normalized)
    elif method == "statistical":
        initialize()
        pieces = jieba.lcut(normalized, HMM=True)
    else:
        raise ValueError(f"Unknown tokenization method: {method}")
    words, positions = [], []
    position = 0
    for piece in pieces:
        piece = piece.strip()
        if not piece or not LEXICAL.search(piece):
            continue
        token = "<num>" if NUMBER.fullmatch(piece) else piece
        if not remove_stopwords or token not in STOPWORDS:
            words.append(token)
            positions.append(position)
        position += 1
    return words, positions


def tokenize(text: str, method: str = "statistical", remove_stopwords: bool = False) -> list[str]:
    return tokenize_with_positions(text, method, remove_stopwords)[0]


def remove_stopwords_from_text(text: str) -> str:
    """Delete statistical-token stopword spans; keep punctuation and word boundaries.

    This operates on text, never on model embeddings. Numeric surfaces are kept
    intact, so callers must apply the same numeric policy in both model modes.
    """
    initialize()
    return "".join(" " if part in STOPWORDS else part
                   for part in jieba.lcut(normalize_text(text), HMM=True))


def prepare(data: Path, output: Path, limit: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    initialize()
    methods = ("matching", "reverse_matching", "statistical")

    prefixes = {
        "matching": "matching_",
        "reverse_matching": "reverse_matching_",
        "statistical": "tokens_",
    }

    files = {
        (method, remove): (output / (
            prefixes[method]
            + ("drop" if remove else "keep")
            + ".jsonl"
        )).open("w", encoding="utf-8")
        for method in methods
        for remove in (False, True)
    }

    comparisons = {
        "fmm_vs_rmm": ("matching", "reverse_matching"),
        "fmm_vs_jieba": ("matching", "statistical"),
        "rmm_vs_jieba": ("reverse_matching", "statistical"),
    }

    pairwise_differing_fields = {
        name: 0 for name in comparisons
    }
    stats = {
        method: {
            "tokenization_seconds": 0.0,
            "tokens_keep": 0,
            "tokens_drop": 0,
            "vocabulary_keep": Counter(),
            "vocabulary_drop": Counter(),
        }
        for method in methods
    }
    examples = []
    doc_ids: set[int] = set()
    title_chars = content_chars = empty_titles = empty_contents = 0
    started = time.perf_counter()
    try:
        with (data / "documents.jsonl").open(encoding="utf-8") as source:
            for line in source:
                if limit is not None and len(doc_ids) >= limit:
                    break
                doc = json.loads(line)
                if doc["doc_id"] in doc_ids:
                    raise ValueError(f"Duplicate doc_id: {doc['doc_id']}")
                doc_ids.add(doc["doc_id"])
                title, content = doc.get("title") or "", doc.get("content") or ""
                title_chars += len(title)
                content_chars += len(content)
                empty_titles += not bool(title.strip())
                empty_contents += not bool(content.strip())
                field_results = {}
                for method in methods:
                    begin = time.perf_counter()
                    title_tokens, title_positions = tokenize_with_positions(title, method)
                    content_tokens, content_positions = tokenize_with_positions(content, method)
                    stats[method]["tokenization_seconds"] += time.perf_counter() - begin
                    field_results[method] = (title_tokens, content_tokens)
                    for remove in (False, True):
                        def select(tokens, positions):
                            return [(token, pos) for token, pos in zip(tokens, positions)
                                    if not remove or token not in STOPWORDS]
                        tt, ct = select(title_tokens, title_positions), select(content_tokens, content_positions)
                        row = {"doc_id": doc["doc_id"], "title": title,
                               "title_tokens": [token for token, _ in tt],
                               "content_tokens": [token for token, _ in ct],
                               "title_positions": [pos for _, pos in tt],
                               "content_positions": [pos for _, pos in ct]}
                        files[method, remove].write(json.dumps(row, ensure_ascii=False) + "\n")
                        key = "drop" if remove else "keep"
                        stats[method]["tokens_" + key] += len(tt) + len(ct)
                        stats[method]["vocabulary_" + key].update(row["title_tokens"] + row["content_tokens"])
                differences = {}
                for name, (left, right) in comparisons.items():
                    count = sum(
                        a != b
                        for a, b in zip(field_results[left], field_results[right])
                    )
                    differences[name] = count
                    pairwise_differing_fields[name] += count
                if any(differences.values()) and len(examples) < 30:
                    example = {"doc_id": doc["doc_id"], "title": title,
                               "content_excerpt": content[:250]}
                    for method in methods:
                        example[f"{method}_title"] = field_results[method][0]
                        example[f"{method}_content"] = field_results[method][1][:70]
                    examples.append(example)
                if len(doc_ids) % 5000 == 0:
                    print(f"Prepared {len(doc_ids):,} documents", flush=True)
    finally:
        for handle in files.values():
            handle.close()
    for method in methods:
        for key in ("keep", "drop"):
            counts = stats[method].pop("vocabulary_" + key)
            stats[method]["vocabulary_" + key] = len(counts)
            stats[method]["top_terms_" + key] = counts.most_common(20)
    summary = {
        "document_count": len(doc_ids), "title_characters": title_chars,
        "content_characters": content_chars, "empty_titles": empty_titles,
        "empty_contents": empty_contents, "total_seconds": time.perf_counter() - started,
        "title_content_policy": "segmented separately; separate positional fields; combined for ranking",
        "matching": "forward maximum matching with Jieba's dictionary; ASCII runs; single-character OOV",
        "reverse_matching": "reverse maximum matching with Jieba's dictionary; ASCII runs; single-character OOV",
        "statistical": "Jieba frequency-weighted DAG decoding, with HMM for unknown Chinese words",
        "normalization": "NFKC; HTML unescape; lowercase; remove [话题]; remove punctuation/emoji-only tokens; pure numeric tokens -> <num>",
        "position_policy": "0-based lexical-token offsets before stopword deletion; title/content independent",
        "stopword_source": "resources/stopwords.txt: manually defined course-experiment list; negations retained",
        "stopword_count": len(STOPWORDS),
        "stopword_sha256": hashlib.sha256(STOPWORD_PATH.read_bytes()).hexdigest(),
        "methods": stats, "comparison_examples": examples,
        "pairwise_differing_fields": pairwise_differing_fields,
    }
    (output / "preprocessing_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    from .data import describe_data
    describe_data(data, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Qilin-IR-50K"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = prepare(args.data, args.output, args.limit)
    print(json.dumps({k: result[k] for k in ("document_count", "total_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
