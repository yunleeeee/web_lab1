"""Field-aware positional index and two independent lossless binary codecs.

Document IDs in the posting arrays are dense ordinals; ``doc_ids`` maps these
back to the original non-contiguous IDs. Positions are lexical-token offsets
before stopword removal. No serialization uses pickle or generic compression.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from typing import Iterable

MAGIC = b"LIRIDX1\n"
U32 = struct.Struct("<I")


@dataclass(slots=True)
class PostingList:
    docs: array
    title_offsets: array
    content_offsets: array
    title_positions: array
    content_positions: array

    @classmethod
    def empty(cls) -> "PostingList":
        return cls(array("I"), array("I", [0]), array("I", [0]), array("I"), array("I"))

    def add(self, ordinal: int, title: Iterable[int], content: Iterable[int]) -> None:
        self.docs.append(ordinal)
        self.title_positions.extend(title)
        self.content_positions.extend(content)
        self.title_offsets.append(len(self.title_positions))
        self.content_offsets.append(len(self.content_positions))

    def positions(self, posting: int, field: str) -> array:
        offsets = self.title_offsets if field == "title" else self.content_offsets
        positions = self.title_positions if field == "title" else self.content_positions
        return positions[offsets[posting]:offsets[posting + 1]]

    def term_frequency(self, posting: int) -> int:
        return (self.title_offsets[posting + 1] - self.title_offsets[posting]
                + self.content_offsets[posting + 1] - self.content_offsets[posting])


@dataclass(slots=True)
class PositionalIndex:
    doc_ids: list[int]
    titles: list[str]
    postings: dict[str, PostingList]
    metadata: dict

    @property
    def document_count(self) -> int:
        return len(self.doc_ids)

    def statistics(self) -> dict:
        posting_count = sum(len(p.docs) for p in self.postings.values())
        position_count = sum(len(p.title_positions) + len(p.content_positions)
                             for p in self.postings.values())
        return {"documents": self.document_count, "terms": len(self.postings),
                "document_term_pairs": posting_count, "positions": position_count,
                "posting_arrays_bytes": sum(sum(len(a) * a.itemsize for a in (
                    p.docs, p.title_offsets, p.content_offsets,
                    p.title_positions, p.content_positions)) for p in self.postings.values())}

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps([self.doc_ids, self.titles], ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
        for term in sorted(self.postings):
            term_bytes = term.encode("utf-8")
            digest.update(U32.pack(len(term_bytes)))
            digest.update(term_bytes)
            p = self.postings[term]
            for a in (p.docs, p.title_offsets, p.content_offsets,
                      p.title_positions, p.content_positions):
                digest.update(U32.pack(len(a)))
                digest.update(_array_bytes(a))
        return digest.hexdigest()


def build_index(token_path: str | Path) -> PositionalIndex:
    started = time.perf_counter()
    doc_ids, titles, postings, seen = [], [], {}, set()
    with Path(token_path).open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            doc_id = int(record["doc_id"])
            if doc_id in seen:
                raise ValueError(f"Duplicate document ID: {doc_id}")
            seen.add(doc_id)
            ordinal = len(doc_ids)
            doc_ids.append(doc_id)
            titles.append(record.get("title") or "")
            by_term: dict[str, tuple[list[int], list[int]]] = {}
            for field_number, field in enumerate(("title", "content")):
                tokens = record[field + "_tokens"]
                positions = record.get(field + "_positions", range(len(tokens)))
                if len(tokens) != len(positions):
                    raise ValueError(f"Position length mismatch in document {doc_id}/{field}")
                previous = -1
                for token, position in zip(tokens, positions):
                    if not isinstance(position, int) or position <= previous or position < 0:
                        raise ValueError(f"Positions must strictly increase: {doc_id}/{field}")
                    previous = position
                    if not token:
                        raise ValueError("Empty indexed term")
                    by_term.setdefault(token, ([], []))[field_number].append(position)
            for term, (title, content) in by_term.items():
                if term not in postings:
                    postings[term] = PostingList.empty()
                postings[term].add(ordinal, title, content)
            if len(doc_ids) % 10000 == 0:
                print(f"Indexed {len(doc_ids):,} documents from {Path(token_path).name}", flush=True)
    return PositionalIndex(doc_ids, titles, postings, {
        "source": str(token_path), "build_seconds": time.perf_counter() - started,
        "position_policy": "zero-based lexical positions before stopword removal; independent fields",
        "document_id_policy": "dense input-order ordinal, mapped back to original doc_id",
    })


def encode_variable_byte(values: Iterable[int]) -> bytes:
    """Seven data bits per byte; high bit marks the LAST byte of a number."""
    output = bytearray()
    for value in values:
        if value < 0:
            raise ValueError("Variable-byte values must be nonnegative")
        while value >= 128:
            output.append(value & 127)
            value >>= 7
        output.append(value | 128)
    return bytes(output)


def decode_variable_byte(data: bytes | memoryview) -> list[int]:
    result, value, shift = [], 0, 0
    for byte in data:
        value |= (byte & 127) << shift
        if byte & 128:
            result.append(value)
            value, shift = 0, 0
        else:
            shift += 7
            if shift > 35:
                raise ValueError("Invalid variable-byte integer")
    if shift:
        raise ValueError("Truncated variable-byte integer")
    return result


def _array_bytes(values: array) -> bytes:
    if sys.byteorder == "little":
        return values.tobytes()
    copy = array("I", values)
    copy.byteswap()
    return copy.tobytes()


def _gap_positions(positions: array, offsets: array) -> Iterable[int]:
    for posting in range(len(offsets) - 1):
        previous = 0
        for i in range(offsets[posting], offsets[posting + 1]):
            position = positions[i]
            yield position - previous
            previous = position


def _write_dictionary(terms: list[str], frontcoding: bool) -> bytes:
    output, previous = bytearray(), b""
    for term in terms:
        encoded = term.encode("utf-8")
        if frontcoding:
            common = 0
            for left, right in zip(previous, encoded):
                if left != right:
                    break
                common += 1
            suffix = encoded[common:]
            output.extend(encode_variable_byte((common, len(suffix))))
            output.extend(suffix)
        else:
            output.extend(U32.pack(len(encoded)))
            output.extend(encoded)
        previous = encoded
    return bytes(output)


def _posting_bytes(p: PostingList, variable_byte: bool) -> bytes:
    if not variable_byte:
        return (struct.pack("<III", len(p.docs), len(p.title_positions), len(p.content_positions))
                + b"".join(_array_bytes(a) for a in (
                    p.docs, p.title_offsets, p.content_offsets,
                    p.title_positions, p.content_positions)))
    def values():
        yield len(p.docs)
        yield len(p.title_positions)
        yield len(p.content_positions)
        previous = 0
        for doc in p.docs:
            yield doc - previous
            previous = doc
        for offsets in (p.title_offsets, p.content_offsets):
            for i in range(len(p.docs)):
                yield offsets[i + 1] - offsets[i]
        yield from _gap_positions(p.title_positions, p.title_offsets)
        yield from _gap_positions(p.content_positions, p.content_offsets)
    return encode_variable_byte(values())


def save_one_index(index: PositionalIndex, path: str | Path,
                   variable_byte: bool = False, frontcoding: bool = False) -> dict:
    started = time.perf_counter()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    terms = sorted(index.postings)
    dictionary = _write_dictionary(terms, frontcoding)
    header = {"version": 1, "variable_byte": variable_byte, "frontcoding": frontcoding,
              "doc_ids": index.doc_ids, "titles": index.titles, "terms": len(terms),
              "dictionary_bytes": len(dictionary), "metadata": index.metadata}
    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as sink:
        sink.write(MAGIC)
        sink.write(U32.pack(len(header_bytes)))
        sink.write(header_bytes)
        sink.write(dictionary)
        postings_start = sink.tell()
        for term in terms:
            sink.write(_posting_bytes(index.postings[term], variable_byte))
        postings_bytes = sink.tell() - postings_start
    return {"path": str(path), "bytes": path.stat().st_size,
            "metadata_bytes": len(MAGIC) + 4 + len(header_bytes),
            "dictionary_bytes": len(dictionary), "postings_bytes": postings_bytes,
            "save_seconds": time.perf_counter() - started,
            "postings_codec": "doc/position gaps + VariableByte" if variable_byte else "uint32 arrays",
            "dictionary_codec": "UTF-8 byte-prefix front coding" if frontcoding else "length + UTF-8"}


def save_index(index: PositionalIndex, output_dir: str | Path, stem: str = "keep") -> dict:
    result = {}
    for name, vb, fc in (("raw", False, False), ("vbyte", True, False),
                         ("frontcoded", False, True), ("combined", True, True)):
        result[name] = save_one_index(index, Path(output_dir) / f"{stem}_{name}.idx", vb, fc)
        result[name]["ratio_to_raw"] = result[name]["bytes"] / result["raw"]["bytes"]
    return result


class _Reader:
    __slots__ = ("data", "offset")

    def __init__(self, data: bytes):
        self.data, self.offset = memoryview(data), 0

    def uint32(self) -> int:
        result = U32.unpack_from(self.data, self.offset)[0]
        self.offset += 4
        return result

    def variable_byte(self) -> int:
        value, shift = 0, 0
        while True:
            if self.offset >= len(self.data):
                raise ValueError("Truncated variable-byte index")
            byte = self.data[self.offset]
            self.offset += 1
            value |= (byte & 127) << shift
            if byte & 128:
                return value
            shift += 7
            if shift > 35:
                raise ValueError("Invalid variable-byte index")

    def take(self, count: int) -> memoryview:
        end = self.offset + count
        if end > len(self.data):
            raise ValueError("Truncated index")
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def uint32_array(self, count: int) -> array:
        result = array("I")
        result.frombytes(self.take(count * 4))
        if sys.byteorder != "little":
            result.byteswap()
        return result


def load_index(path: str | Path) -> PositionalIndex:
    started = time.perf_counter()
    with Path(path).open("rb") as source:
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError("Invalid index signature")
        header_size_bytes = source.read(4)
        if len(header_size_bytes) != 4:
            raise ValueError("Truncated index header")
        header = json.loads(source.read(U32.unpack(header_size_bytes)[0]))
        dictionary = _Reader(source.read(header["dictionary_bytes"]))
        posting_data = _Reader(source.read())
    terms, previous = [], b""
    for _ in range(header["terms"]):
        if header["frontcoding"]:
            common, length = dictionary.variable_byte(), dictionary.variable_byte()
            if common > len(previous):
                raise ValueError("Invalid front-coded prefix")
            encoded = previous[:common] + bytes(dictionary.take(length))
        else:
            encoded = bytes(dictionary.take(dictionary.uint32()))
        terms.append(encoded.decode("utf-8"))
        previous = encoded
    if dictionary.offset != len(dictionary.data) or len(set(terms)) != len(terms):
        raise ValueError("Invalid index dictionary")
    postings = {}
    number = posting_data.variable_byte if header["variable_byte"] else posting_data.uint32
    for term in terms:
        df, title_size, content_size = number(), number(), number()
        if not header["variable_byte"]:
            p = PostingList(*(posting_data.uint32_array(n) for n in
                              (df, df + 1, df + 1, title_size, content_size)))
        else:
            docs, current = array("I"), 0
            for _ in range(df):
                current += number()
                docs.append(current)
            offsets = []
            for _ in range(2):
                field_offsets, current = array("I", [0]), 0
                for _ in range(df):
                    current += number()
                    field_offsets.append(current)
                offsets.append(field_offsets)
            positions = []
            for field_offsets in offsets:
                field_positions = array("I")
                for posting in range(df):
                    current = 0
                    for _ in range(field_offsets[posting + 1] - field_offsets[posting]):
                        current += number()
                        field_positions.append(current)
                positions.append(field_positions)
            p = PostingList(docs, *offsets, *positions)
        if (p.title_offsets[-1] != title_size or p.content_offsets[-1] != content_size
                or (df and p.docs[-1] >= len(header["doc_ids"]))):
            raise ValueError("Invalid posting sizes or document ordinal")
        postings[term] = p
    if posting_data.offset != len(posting_data.data):
        raise ValueError("Trailing or unparsed posting bytes")
    metadata = dict(header.get("metadata", {}))
    metadata.update({"loaded_from": str(path), "load_decode_seconds": time.perf_counter() - started,
                     "codec": {"variable_byte": header["variable_byte"], "frontcoding": header["frontcoding"]}})
    return PositionalIndex(header["doc_ids"], header["titles"], postings, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/indexes"))
    parser.add_argument("--stem", default="keep")
    args = parser.parse_args()
    index = build_index(args.tokens)
    print(json.dumps({"statistics": index.statistics(), "files": save_index(index, args.output, args.stem)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
