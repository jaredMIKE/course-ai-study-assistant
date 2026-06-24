from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

STOPWORDS = set(
    "的 了 和 是 在 与 对 中 为 及 或 一个 一种 进行 实现 基于 通过 可以 系统 用户 资料 课程 模块 内容 需要 包括 当前 这个 那个 以及 并且 其中 主要 相关 具有 使用 利用 支持 完成 提供".split()
)


def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t\u3000]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """轻量中文检索分词：中文二元/三元词片 + 英文单词。无需额外安装分词库，便于本地跑。"""
    cn_blocks = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
    en_words = re.findall(r"[a-zA-Z0-9_+#.-]{2,}", text.lower())
    words: list[str] = []
    for block in cn_blocks:
        words.extend(block[i : i + 2] for i in range(max(0, len(block) - 1)))
        words.extend(block[i : i + 3] for i in range(max(0, len(block) - 2)))
    return [word for word in words + en_words if word and word not in STOPWORDS]


def keywords(text: str, limit: int = 10) -> list[str]:
    return [word for word, _ in Counter(tokenize(text)).most_common(limit)]


def split_text(text: str, size: int = 650, overlap: int = 120) -> list[str]:
    """段落优先分块，长段落再按窗口切分；保留重叠上下文。"""
    clean = normalize(text)
    if not clean:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            units.append(paragraph)
            continue
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])", paragraph) if s.strip()]
        if len(sentences) <= 1:
            step = max(100, size - overlap)
            units.extend(paragraph[i : i + size] for i in range(0, len(paragraph), step))
        else:
            units.extend(sentences)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= size:
            current = candidate
        else:
            if len(current) >= 10:
                chunks.append(current)
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                current = f"{tail}\n{unit}".strip()
            else:
                current = unit
    if len(current) >= 10:
        chunks.append(current)
    return chunks


def sentence_extract(text: str, limit: int = 6) -> list[str]:
    sentences = [part.strip() for part in re.split(r"[。！？!?\n]", normalize(text))]
    return [item for item in sentences if len(item) > 10][:limit]


def tfidf_vectors(texts: list[str]) -> tuple[list[Counter], dict[str, float]]:
    token_lists = [tokenize(text) for text in texts]
    n = max(1, len(token_lists))
    df: Counter[str] = Counter()
    for tokens in token_lists:
        df.update(set(tokens))
    idf = {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}
    vectors: list[Counter] = []
    for tokens in token_lists:
        tf = Counter(tokens)
        vec = Counter({term: value * idf.get(term, 1.0) for term, value in tf.items()})
        vectors.append(vec)
    return vectors, idf


def vectorize_with_idf(text: str, idf: dict[str, float]) -> Counter:
    tf = Counter(tokenize(text))
    return Counter({term: value * idf.get(term, 1.0) for term, value in tf.items()})


def cosine(a: Counter, b: Counter) -> float:
    norm_a = sum(value * value for value in a.values())
    norm_b = sum(value * value for value in b.values())
    if not norm_a or not norm_b:
        return 0.0
    # 遍历较短向量，提升速度。
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def compact_context(sources: list[dict], max_chars: int = 5500) -> str:
    parts: list[str] = []
    total = 0
    for i, item in enumerate(sources, start=1):
        text = item["text"].strip()
        part = f"[来源{i}] 文档：{item['document_title']}；片段：{item['chunk_index']}；相似度：{item['score']:.4f}\n{text}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n".join(parts)
